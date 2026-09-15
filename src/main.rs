#[global_allocator]
static GLOBAL: mimalloc::MiMalloc = mimalloc::MiMalloc;

use ahash::RandomState;
use ark_ec::{CurveGroup, Group};
use ark_ff::{BigInteger, Field, One, PrimeField, Zero};
use ark_secp256k1::{Affine, Fq, Fr, Projective};
use bs58;
use clap::Parser;
use crc32fast::Hasher;
use crossbeam_channel::{bounded, Sender};
use ctrlc;
use hashbrown::HashMap;
use hex;
use rand::{Rng, SeedableRng};
use rand_xoshiro::Xoshiro256PlusPlus;
use ripemd::Ripemd160;
use sha2::{Digest, Sha256};
use serde::{Serialize, Deserialize};
use std::{
    fs::{self, File, OpenOptions},
    io::{Read, BufWriter, Write, ErrorKind},
    sync::{atomic::{AtomicBool, AtomicU64, Ordering}, Arc},
    thread, time::{Duration, Instant, SystemTime, UNIX_EPOCH},
};

const FOUND_FILE: &str = "kangputus_found.txt";
const NUM_JUMPS: usize = 128;
const JUMP_MASK: usize = 127;
const K_BATCH: usize = 2048; 
const RECORD_SIZE: usize = 77; 
const WINDOW_SIZE: usize = 8;

static KEEP_RUNNING: AtomicBool = AtomicBool::new(true);
static ALREADY_FOUND: AtomicBool = AtomicBool::new(false);

// =========================================================================
// KHAI BÁO LIÊN KẾT FFI ĐẾN HẠT NHÂN C++ MONTGOMERY
// =========================================================================
extern "C" {
    fn c_point_update(
        px: *const u64, py: *const u64,
        qx: *const u64, qy: *const u64,
        inv: *const u64,
        rx: *mut u64, ry: *mut u64,
        batch_size: usize
    );
}
// =========================================================================

#[derive(Serialize, Deserialize)]
struct Manifest {
    version: u32,
    checkpoint_gen: u64,
    records_total: u64, 
    last_seq: u64,
    size: u64,
    status: String,
}

#[derive(Hash, PartialEq, Eq, Clone, Copy)]
struct DpKey { x_0: u64, x_1: u64, x_2: u64, x_3: u64 }

#[derive(Clone)]
struct DpMessage { target_coeff: i8, key: DpKey, scalar_limbs: [u64; 4] }

struct WorkerState { keys_scanned: AtomicU64 }

enum DiskMsg {
    Record([u8; RECORD_SIZE]), 
    LocalFlushReq,
    CheckpointBarrier { gen: u64, is_final: bool },
}

#[inline(always)]
fn batch_inversion_in_place(v: &mut [Fq], scratch: &mut [Fq]) {
    let n = v.len();
    if n == 0 { return; }
    scratch[0] = v[0];
    for i in 1..n { scratch[i] = scratch[i - 1] * v[i]; }
    let mut inv = match scratch[n - 1].inverse() { Some(val) => val, None => Fq::one() };
    for i in (1..n).rev() { let tmp = scratch[i - 1] * inv; inv = inv * v[i]; v[i] = tmp; }
    v[0] = inv;
}

struct FixedBase { window_size: usize, num_windows: usize, table: Vec<Vec<Affine>> }
impl FixedBase {
    fn new(window_size: usize) -> Self {
        let num_windows = (256 + window_size - 1) / window_size;
        let table_size = 1 << window_size;
        let mut table = Vec::with_capacity(num_windows);
        let mut base = Projective::generator();
        for _ in 0..num_windows {
            let mut win_table = Vec::with_capacity(table_size);
            let mut current = Projective::zero();
            for _ in 0..table_size { win_table.push(current); current += base; }
            table.push(Projective::normalize_batch(&win_table));
            for _ in 0..window_size { base.double_in_place(); }
        }
        Self { window_size, num_windows, table }
    }
    #[inline(always)]
    fn mul(&self, scalar: &Fr) -> Projective {
        let limbs = scalar.into_bigint().0;
        let mut res = Projective::zero();
        let mask = (1 << self.window_size) - 1;
        for win in 0..self.num_windows {
            let bit_offset = win * self.window_size;
            let limb_idx = bit_offset / 64;
            let bit_shift = bit_offset % 64;
            let val = if limb_idx < 4 {
                let mut raw = limbs[limb_idx] >> bit_shift;
                if bit_shift > (64 - self.window_size) && limb_idx + 1 < 4 { raw |= limbs[limb_idx + 1] << (64 - bit_shift); }
                (raw as usize) & mask
            } else { 0 };
            if val > 0 { res += self.table[win][val]; }
        }
        res
    }
}

#[derive(Parser, Debug)]
#[command(author, version, about)]
struct Args {
    #[arg(short, long)] cores: Option<usize>,
    #[arg(short, long)] target: String,
    #[arg(short, long)] start: String,
    #[arg(short, long)] end: String,
    #[arg(long, default_value = "26")] dp_bits: u32,
    #[arg(long, default_value = "139")] sub_bits: u32,
    #[arg(long)] local_dir: String,
    #[arg(long)] drive_dir: String,
}

#[inline(always)] fn scalar_to_limbs(scalar: Fr) -> [u64; 4] { scalar.into_bigint().0 }
#[inline(always)] fn limbs_to_fr(limbs: [u64; 4]) -> Fr { Fr::from_bigint(ark_ff::BigInt(limbs)).unwrap() }
#[inline(always)] fn scalar_to_bytes(scalar: Fr) -> [u8; 32] {
    let limbs = scalar.into_bigint().0; let mut out = [0u8; 32];
    out[0..8].copy_from_slice(&limbs[3].to_be_bytes()); out[8..16].copy_from_slice(&limbs[2].to_be_bytes());
    out[16..24].copy_from_slice(&limbs[1].to_be_bytes()); out[24..32].copy_from_slice(&limbs[0].to_be_bytes());
    out
}

fn compact_u128(value: u128) -> String {
    const UNITS: &[(u128, &str)] = &[(1_000_000_000_000_000, "Q"), (1_000_000_000_000, "T"), (1_000_000_000, "B"), (1_000_000, "M"), (1_000, "K")];
    for &(divisor, suffix) in UNITS { if value >= divisor { return format!("{:.2}{}", value as f64 / divisor as f64, suffix); } }
    value.to_string()
}

fn parse_hex_to_fr(hex_str: &str) -> Fr {
    let clean = hex_str.trim_start_matches("0x").trim_start_matches("0X");
    let decoded = hex::decode(if clean.len() % 2 != 0 { format!("0{}", clean) } else { clean.to_string() }).unwrap();
    let mut bytes = [0u8; 32]; bytes[32 - decoded.len()..].copy_from_slice(&decoded);
    Fr::from_be_bytes_mod_order(&bytes)
}

fn make_fast_rng(worker_id: usize) -> Xoshiro256PlusPlus {
    let nanos = SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_nanos();
    let mut seed = [0u8; 32];
    seed[0..16].copy_from_slice(&nanos.to_le_bytes()); seed[16..24].copy_from_slice(&(std::process::id() as u64).to_le_bytes());
    seed[24..32].copy_from_slice(&(worker_id as u64).to_le_bytes());
    Xoshiro256PlusPlus::from_seed(Sha256::digest(seed).into())
}

fn random_fr_below(rng: &mut Xoshiro256PlusPlus, upper: Fr) -> Fr {
    if upper.is_zero() { return Fr::zero(); }
    let upper_big = upper.into_bigint();
    let bits = upper_big.num_bits();
    loop {
        let mut rand_bytes = [0u8; 32]; rng.fill(&mut rand_bytes);
        let mut current_bit = 0;
        for i in (0..32).rev() { for bit in 0..8 { if current_bit >= bits { rand_bytes[i] &= !(1u8 << bit); } current_bit += 1; } }
        let candidate = Fr::from_be_bytes_mod_order(&rand_bytes);
        if candidate.into_bigint() < upper_big { return candidate; }
    }
}

fn worker_loop(
    start_as_base: bool, jump_scalars: &[Fr], jump_points: &[Affine], target_projective: Projective,
    fixed_base: &FixedBase, state: &WorkerState, tx_dp: Sender<DpMessage>,
    range_start: Fr, range_width: Fr, dp_bits: u32, sub_bits: u32, worker_id: usize
) {
    let mut rng = make_fast_rng(worker_id);
    let mut p_x = vec![Fq::zero(); K_BATCH];
    let mut p_y = vec![Fq::zero(); K_BATCH];
    let mut scalars = vec![Fr::zero(); K_BATCH];
    let mut target_coeffs = vec![0i8; K_BATCH]; 
    let mut walked_dist = vec![Fr::zero(); K_BATCH];  
    
    let mut jump_indices = vec![0usize; K_BATCH];
    let mut denominators = vec![Fq::one(); K_BATCH];
    let mut scratch_pad = vec![Fq::zero(); K_BATCH];
    let mut p_x_limbs_cache = vec![[0u64; 4]; K_BATCH];

    // Các mảng phẳng chuẩn bị cho C++
    let mut q_x = vec![Fq::zero(); K_BATCH];
    let mut q_y = vec![Fq::zero(); K_BATCH];
    let mut r_x = vec![Fq::zero(); K_BATCH];
    let mut r_y = vec![Fq::zero(); K_BATCH];

    let max_distance = if sub_bits >= 255 { Fr::from(-1i8) } else { Fr::from(2u64).pow([sub_bits as u64]) };

    let mut respawn_point = |i: usize, rng: &mut Xoshiro256PlusPlus, p_x: &mut [Fq], p_y: &mut [Fq], sc: &mut [Fr], tc: &mut [i8], dist: &mut [Fr], p_x_limbs_cache: &mut [[u64; 4]]| {
        if start_as_base {
            let s_t = range_start + random_fr_below(rng, range_width);
            let affine_pt = fixed_base.mul(&s_t).into_affine();
            p_x[i] = affine_pt.x; p_y[i] = affine_pt.y; sc[i] = s_t; tc[i] = 0;
            p_x_limbs_cache[i] = affine_pt.x.into_bigint().0;
        } else {
            let wild_radius = Fr::from(2u64).pow([(sub_bits / 2) as u64]);
            let s_w = random_fr_below(rng, wild_radius + wild_radius) - wild_radius;
            let affine_pt = (target_projective + fixed_base.mul(&s_w)).into_affine();
            p_x[i] = affine_pt.x; p_y[i] = affine_pt.y; sc[i] = s_w; tc[i] = 1;
            p_x_limbs_cache[i] = affine_pt.x.into_bigint().0;
        }
        dist[i] = Fr::zero(); 
    };

    for i in 0..K_BATCH { respawn_point(i, &mut rng, &mut p_x, &mut p_y, &mut scalars, &mut target_coeffs, &mut walked_dist, &mut p_x_limbs_cache); }
    let mut local_counter = 0u64;

    while KEEP_RUNNING.load(Ordering::Relaxed) {
        for i in 0..K_BATCH {
            let mut x_limbs = p_x_limbs_cache[i];
            let mut mix = x_limbs[0] ^ x_limbs[1].rotate_left(17) ^ x_limbs[2].rotate_left(33) ^ x_limbs[3].rotate_left(51);
            jump_indices[i] = (mix as usize) & JUMP_MASK;
            
            let pt = &jump_points[jump_indices[i]];
            q_x[i] = pt.x;
            q_y[i] = pt.y;
            denominators[i] = pt.x - p_x[i];
            
            while denominators[i].is_zero() { 
                respawn_point(i, &mut rng, &mut p_x, &mut p_y, &mut scalars, &mut target_coeffs, &mut walked_dist, &mut p_x_limbs_cache);
                x_limbs = p_x_limbs_cache[i];
                mix = x_limbs[0] ^ x_limbs[1].rotate_left(17) ^ x_limbs[2].rotate_left(33) ^ x_limbs[3].rotate_left(51);
                jump_indices[i] = (mix as usize) & JUMP_MASK;
                
                let pt = &jump_points[jump_indices[i]];
                q_x[i] = pt.x;
                q_y[i] = pt.y;
                denominators[i] = pt.x - p_x[i];
            }
        }

        // Rust xử lý Batch Inversion
        batch_inversion_in_place(&mut denominators, &mut scratch_pad);
        let mut actual_hops = 0u64;

        // [KÍCH HOẠT VŨ KHÍ C++]
        // Truyền thẳng 5 mảng phẳng vào lõi C++ để tính toàn bộ X_new và Y_new cùng lúc
        unsafe {
            c_point_update(
                p_x.as_ptr() as *const u64, p_y.as_ptr() as *const u64,
                q_x.as_ptr() as *const u64, q_y.as_ptr() as *const u64,
                denominators.as_ptr() as *const u64,
                r_x.as_mut_ptr() as *mut u64, r_y.as_mut_ptr() as *mut u64,
                K_BATCH
            );
        }

        // Lấy kết quả từ C++ và xử lý DP
        for i in 0..K_BATCH {
            let j_idx = jump_indices[i];
            
            p_x[i] = r_x[i]; 
            p_y[i] = r_y[i]; 
            
            scalars[i] = scalars[i] + jump_scalars[j_idx];
            let new_limbs = p_x[i].into_bigint().0;
            p_x_limbs_cache[i] = new_limbs;
            actual_hops += 1;
            
            walked_dist[i] += jump_scalars[j_idx];

            if new_limbs[0].trailing_zeros() >= dp_bits {
                let msg = DpMessage {
                    target_coeff: target_coeffs[i],
                    key: DpKey { x_0: new_limbs[0], x_1: new_limbs[1], x_2: new_limbs[2], x_3: new_limbs[3] },
                    scalar_limbs: scalar_to_limbs(scalars[i])
                };
                
                let mut sent = false;
                while !sent && KEEP_RUNNING.load(Ordering::Relaxed) {
                    match tx_dp.send_timeout(msg.clone(), Duration::from_millis(50)) {
                        Ok(_) => sent = true,
                        Err(crossbeam_channel::SendTimeoutError::Timeout(_)) => continue,
                        Err(crossbeam_channel::SendTimeoutError::Disconnected(_)) => break,
                    }
                }
                respawn_point(i, &mut rng, &mut p_x, &mut p_y, &mut scalars, &mut target_coeffs, &mut walked_dist, &mut p_x_limbs_cache);
            } else if walked_dist[i].into_bigint() > max_distance.into_bigint() {
                respawn_point(i, &mut rng, &mut p_x, &mut p_y, &mut scalars, &mut target_coeffs, &mut walked_dist, &mut p_x_limbs_cache);
            }
        }

        local_counter += actual_hops;
        if local_counter >= 16_384 { 
            state.keys_scanned.fetch_add(local_counter, Ordering::Relaxed); 
            local_counter = 0; 
        }
    }
}

fn validate_and_load_db(path: &str, map: &mut HashMap<DpKey, Vec<(i8, [u64; 4])>, RandomState>, expected_start_seq: &mut u64) -> Result<(u64, u64), String> {
    let mut count = 0;
    let file = match File::open(path) {
        Ok(f) => f,
        Err(e) if e.kind() == ErrorKind::NotFound => return Ok((0, 0)),
        Err(e) => return Err(format!("I/O Error: {}", e)),
    };

    let mut buf = [0u8; RECORD_SIZE];
    let mut current_expected_seq = *expected_start_seq;
    let mut reader = std::io::BufReader::new(file);
    let mut last_seq_read = 0;
    
    loop {
        let mut bytes_read = 0;
        while bytes_read < RECORD_SIZE {
            match reader.read(&mut buf[bytes_read..]) {
                Ok(0) => {
                    if bytes_read == 0 { 
                        *expected_start_seq = current_expected_seq;
                        return Ok((count, last_seq_read)); 
                    } else {
                        return Err(format!("File bị cụt tại byte thứ {}", bytes_read));
                    }
                }
                Ok(n) => bytes_read += n,
                Err(ref e) if e.kind() == ErrorKind::Interrupted => continue,
                Err(e) => return Err(format!("Lỗi HĐH đọc file: {}", e)),
            }
        }

        let mut hasher = Hasher::new(); hasher.update(&buf[0..73]);
        if hasher.finalize() != u32::from_be_bytes(buf[73..77].try_into().unwrap()) {
            return Err(format!("CRC32 hỏng tại record {}", count));
        }

        let seq = u64::from_be_bytes(buf[0..8].try_into().unwrap());
        if seq != current_expected_seq {
            return Err(format!("SEQ nhảy cóc (Mong đợi {}, Nhận {}). Corrupted!", current_expected_seq, seq));
        }
        
        last_seq_read = seq;
        current_expected_seq = seq + 1;

        let mut x = [0u64; 4]; let mut s = [0u64; 4];
        for i in 0..4 {
            x[i] = u64::from_be_bytes(buf[8 + i*8 .. 16 + i*8].try_into().unwrap());
            s[3-i] = u64::from_be_bytes(buf[41 + i*8 .. 49 + i*8].try_into().unwrap());
        }

        let target_coeff = buf[40] as i8;
        map.entry(DpKey { x_0: x[3], x_1: x[2], x_2: x[1], x_3: x[0] }).or_default().push((target_coeff, s));
        count += 1;
    }
}

fn get_all_checkpoints(drive_dir: &str) -> Vec<u64> {
    let mut gens = Vec::new();
    if let Ok(entries) = fs::read_dir(drive_dir) {
        for e in entries.flatten() {
            let name = e.file_name().into_string().unwrap_or_default();
            if name.starts_with("checkpoint_") && !name.ends_with("_tmp") {
                if let Ok(gen) = name.strip_prefix("checkpoint_").unwrap_or("").parse::<u64>() {
                    gens.push(gen);
                }
            }
        }
    }
    gens.sort_unstable();
    gens
}

fn hash160_to_address(hash160: &[u8; 20]) -> String {
    let mut payload = [0u8; 25]; payload[0] = 0x00; payload[1..21].copy_from_slice(hash160);
    let c1 = Sha256::digest(&payload[..21]); payload[21..25].copy_from_slice(&Sha256::digest(c1)[..4]);
    bs58::encode(payload).into_string()
}
fn private_key_to_wif_compressed(priv_key: &[u8; 32]) -> String {
    let mut payload = [0u8; 38]; payload[0] = 0x80; payload[1..33].copy_from_slice(priv_key); payload[33] = 0x01;
    let c1 = Sha256::digest(&payload[..34]); payload[34..38].copy_from_slice(&Sha256::digest(c1)[..4]);
    bs58::encode(payload).into_string()
}

pub fn send_telegram_alert(address: &str, wif: &str, hex: &str) {
    let bot_token = "BOT_TOKEN_CUA_BAN_DIEN_VAO_DAY";
    let chat_id = "CHAT_ID_CUA_BAN_DIEN_VAO_DAY";
    if bot_token.contains("CUA_BAN_DIEN") { return; }
    
    let message = format!("✅ BINGO! MATCH FOUND (140-BIT)!\\n\\nAddress: {}\\nWIF: {}\\nHEX: {}", address, wif, hex);
    let payload = format!("{{\"chat_id\": \"{}\", \"text\": \"{}\"}}", chat_id, message);
    let url = format!("https://api.telegram.org/bot{}/sendMessage", bot_token);
    let _ = std::process::Command::new("curl").arg("-s").arg("-X").arg("POST").arg(&url).arg("-H").arg("Content-Type: application/json").arg("-d").arg(&payload).status();
}

fn verify_and_save(final_scalar: Fr, target_bytes: &[u8; 33], fixed_base: &FixedBase) {
    let priv_bytes = scalar_to_bytes(final_scalar);
    let derived_pubkey_affine = fixed_base.mul(&final_scalar).into_affine();
    let mut derived_bytes = [0u8; 33];
    derived_bytes[0] = if (derived_pubkey_affine.y.into_bigint().0[0] & 1) != 0 { 0x03 } else { 0x02 };
    let x_limbs = derived_pubkey_affine.x.into_bigint().0;
    derived_bytes[1..9].copy_from_slice(&x_limbs[3].to_be_bytes()); derived_bytes[9..17].copy_from_slice(&x_limbs[2].to_be_bytes());
    derived_bytes[17..25].copy_from_slice(&x_limbs[1].to_be_bytes()); derived_bytes[25..33].copy_from_slice(&x_limbs[0].to_be_bytes());

    if &derived_bytes != target_bytes { return; }
    if ALREADY_FOUND.swap(true, Ordering::SeqCst) { return; }

    let mut sha = Sha256::new(); let mut rip = Ripemd160::new();
    sha.update(derived_bytes); rip.update(sha.finalize());
    let mut hash160 = [0u8; 20]; hash160.copy_from_slice(&rip.finalize());

    let addr = hash160_to_address(&hash160); let wif = private_key_to_wif_compressed(&priv_bytes);
    let hex_priv = hex::encode(&priv_bytes);

    let msg = format!("\n======================================\n🚀 BINGO! PUZZLE SOLVED! 🚀\nAddress: {}\nWIF: {}\nHEX: {}\n======================================\n", addr, wif, hex_priv.trim_start_matches('0'));
    println!("{}", msg);
    std::io::stdout().flush().unwrap();
    
    if let Ok(mut f) = OpenOptions::new().create(true).append(true).open(FOUND_FILE) { let _ = f.write_all(msg.as_bytes()); }
    
    KEEP_RUNNING.store(false, Ordering::Release);
    send_telegram_alert(&addr, &wif, hex_priv.trim_start_matches('0'));
}

fn main() {
    let args = Args::parse();
    let active_cores = args.cores.unwrap_or_else(|| std::thread::available_parallelism().map(|n| n.get()).unwrap_or(1));

    let clean_target = args.target.trim_start_matches("0x").trim_start_matches("0X");
    let decoded_target = hex::decode(clean_target).unwrap();
    let mut target_bytes = [0u8; 33]; target_bytes.copy_from_slice(&decoded_target);

    let mut x_bytes = [0u8; 32]; x_bytes.copy_from_slice(&target_bytes[1..33]);
    let mut out_limbs = [0u64; 4]; for i in 0..4 { out_limbs[i] = u64::from_be_bytes(x_bytes[(3-i)*8..(3-i)*8+8].try_into().unwrap()); }
    let x_fq = Fq::from_bigint(ark_ff::BigInt(out_limbs)).unwrap();
    let y_sq = (x_fq * x_fq * x_fq) + Fq::from(7u64); let y_fq = y_sq.sqrt().unwrap();
    let y = if ((y_fq.into_bigint().0[0] & 1) != 0) == (target_bytes[0] == 0x03) { y_fq } else { -y_fq };
    let target_projective = Projective::from(Affine::new_unchecked(x_fq, y));

    let master_start = parse_hex_to_fr(&args.start);
    let master_end = parse_hex_to_fr(&args.end);
    let epoch_delta = if args.sub_bits >= 255 { Fr::from(-1i8) } else { Fr::from(2u64).pow([args.sub_bits as u64]) };

    let safe_master_end = if master_end.into_bigint() > epoch_delta.into_bigint() { master_end - epoch_delta + Fr::one() } else { master_end };
    let master_span = if safe_master_end.into_bigint() <= master_start.into_bigint() { Fr::zero() } else { safe_master_end - master_start };

    let fixed_base = Arc::new(FixedBase::new(WINDOW_SIZE));
    let target_jump_bits = args.sub_bits as f64 / 2.0;
    let mut init_rng = make_fast_rng(999);
    let mut jump_scalars = Vec::with_capacity(NUM_JUMPS);
    let mut jump_points_proj = Vec::with_capacity(NUM_JUMPS);

    for i in 0..NUM_JUMPS {
        let base_jump_bits = if target_jump_bits > 4.0 { target_jump_bits - 4.0 } else { 1.0 };
        let dynamic_bits = base_jump_bits + (8.0 * ((i as f64) / (NUM_JUMPS as f64 - 1.0)));
        let variance = random_fr_below(&mut init_rng, Fr::from(2u64).pow([dynamic_bits.round() as u64]));
        let j_scalar = Fr::from(2u64).pow([(dynamic_bits.round() as u32 - 1) as u64]) + variance;
        jump_scalars.push(j_scalar); jump_points_proj.push(fixed_base.mul(&j_scalar));
    }
    let jump_points = Projective::normalize_batch(&jump_points_proj);

    ctrlc::set_handler(move || { KEEP_RUNNING.store(false, Ordering::Release); }).unwrap();

    let mut dp_map = HashMap::<DpKey, Vec<(i8, [u64; 4])>, RandomState>::with_capacity_and_hasher(2_000_000, RandomState::new());

    let drive_dir = args.drive_dir.clone();
    let checkpoints = get_all_checkpoints(&drive_dir);
    let local_dir = args.local_dir.clone();
    
    let mut active_gen = 0;
    let mut global_seq = 0u64;
    let mut durable_records_count = 0u64;

    println!("[*] Đang nạp Checkpoint từ Google Drive...");
    let mut loaded_successfully = false;

    let _ = fs::remove_dir_all(&local_dir);
    let _ = fs::create_dir_all(&local_dir);

    for &gen in checkpoints.iter().rev() {
        println!("[*] Đánh giá Checkpoint {}...", gen);
        let cp_manifest = format!("{}/checkpoint_{}/manifest.json", drive_dir, gen);
        if !std::path::Path::new(&cp_manifest).exists() { continue; }

        let manifest_str = fs::read_to_string(&cp_manifest).unwrap_or_default();
        if let Ok(manifest) = serde_json::from_str::<Manifest>(&manifest_str) {
            
            if manifest.version != 1 || manifest.status != "committed" || manifest.checkpoint_gen != gen { continue; }

            let cp_db = format!("{}/checkpoint_{}/vow.dp", drive_dir, gen);
            let local_db = format!("{}/vow.dp", local_dir);
            let db_size = fs::metadata(&cp_db).map(|m| m.len()).unwrap_or(0);
            
            if db_size != manifest.size { continue; }

            if fs::copy(&cp_db, &local_db).is_err() { continue; }

            dp_map.clear();
            let mut db_seq = 0;

            match validate_and_load_db(&local_db, &mut dp_map, &mut db_seq) {
                Ok((cnt, last)) => {
                    if cnt == manifest.records_total && (cnt == 0 || last == manifest.last_seq) {
                        durable_records_count = cnt;
                        global_seq = db_seq;
                        active_gen = gen + 1; 
                        loaded_successfully = true;
                        println!("[+] Validated Rollback: Checkpoint {} hoàn hảo! (Tổng DP: {})", gen, cnt);
                        break;
                    }
                },
                Err(e) => { eprintln!("[-] Checkpoint {} lỗi: {}", gen, e); }
            }
        }
    }

    if !loaded_successfully {
        println!("[!] Không có dữ liệu hợp lệ trên Drive. Khởi tạo Database mới hoàn toàn.");
        let _ = fs::remove_dir_all(&local_dir); let _ = fs::create_dir_all(&local_dir);
        dp_map.clear();
        global_seq = 0; durable_records_count = 0; active_gen = 1;
    }

    let mut worker_states = Vec::with_capacity(active_cores);
    for _ in 0..active_cores { worker_states.push(WorkerState { keys_scanned: AtomicU64::new(0) }); }

    println!("\n[+] Kích hoạt {} Lõi (v8.0.0 C++ BMI2 ACCELERATED)...", active_cores);
    std::io::stdout().flush().unwrap();

    let (tx_disk, rx_disk) = bounded::<DiskMsg>(300_000);
    let (tx_ack, rx_ack) = bounded::<(u64, bool)>(1);
    
    let disk_handle = thread::spawn(move || {
        let db_local = format!("{}/vow.dp", local_dir);
        let d_file = match OpenOptions::new().create(true).append(true).open(&db_local) {
            Ok(f) => f, Err(e) => { eprintln!("🔥 FATAL: Cannot open DB: {}", e); std::process::exit(1); }
        };
        
        let mut db_writer = BufWriter::new(d_file);
        let mut disk_seq = global_seq;
        let mut disk_failed = false;

        while let Ok(msg) = rx_disk.recv() {
            match msg {
                DiskMsg::Record(buf) => {
                    if disk_failed { continue; }
                    if db_writer.write_all(&buf).is_err() { 
                        disk_failed = true; KEEP_RUNNING.store(false, Ordering::Release); 
                    } else { disk_seq += 1; }
                },
                DiskMsg::LocalFlushReq => {
                    if disk_failed { continue; }
                    if db_writer.flush().is_err() || db_writer.get_ref().sync_all().is_err() {
                        disk_failed = true; KEEP_RUNNING.store(false, Ordering::Release);
                    }
                },
                DiskMsg::CheckpointBarrier { gen, is_final } => {
                    if disk_failed {
                        if is_final { let _ = tx_ack.send((gen, false)); break; } continue;
                    }

                    while let Ok(msg_drain) = rx_disk.try_recv() {
                        if let DiskMsg::Record(buf) = msg_drain {
                            if db_writer.write_all(&buf).is_err() { disk_failed = true; } else { disk_seq += 1; }
                        }
                    }

                    if disk_failed || db_writer.flush().is_err() || db_writer.get_ref().sync_all().is_err() { 
                        disk_failed = true; KEEP_RUNNING.store(false, Ordering::Release);
                        if is_final { let _ = tx_ack.send((gen, false)); break; } continue; 
                    }

                    let d_size = fs::metadata(&db_local).map(|m| m.len()).unwrap_or(0);
                    let cp_tmp = format!("{}/checkpoint_tmp", drive_dir);
                    let cp_final_dir = format!("{}/checkpoint_{}", drive_dir, gen);
                    
                    let _ = fs::remove_dir_all(&cp_tmp);
                    if fs::create_dir_all(&cp_tmp).is_ok() {
                        let dst_db = format!("{}/vow.dp", cp_tmp);
                        if fs::copy(&db_local, &dst_db).is_ok() {
                            let candidate_records = d_size / (RECORD_SIZE as u64);
                            let manifest = Manifest {
                                version: 1, checkpoint_gen: gen, records_total: candidate_records,
                                last_seq: disk_seq.saturating_sub(1), size: d_size, status: "committed".to_string(),
                            };
                            let manifest_str = serde_json::to_string_pretty(&manifest).unwrap();
                            if fs::write(format!("{}/manifest.json", cp_tmp), manifest_str).is_ok() && fs::rename(&cp_tmp, &cp_final_dir).is_ok() {
                                println!("[*] Checkpoint {} Committed. (Drive an toàn)", gen);
                                if gen > 3 { let _ = fs::remove_dir_all(&format!("{}/checkpoint_{}", drive_dir, gen - 3)); }
                                let _ = tx_ack.send((gen, true));
                                if is_final { break; } continue;
                            }
                        }
                    }
                    let _ = tx_ack.send((gen, false)); if is_final { break; }
                }
            }
        }
        if !disk_failed { let _ = db_writer.flush(); let _ = db_writer.get_ref().sync_all(); }
    });

    thread::scope(|s| {
        let (tx_dp, rx_dp) = bounded::<DpMessage>(300_000);
        
        for idx in 0..active_cores {
            let state = &worker_states[idx]; let tx_dp_clone = tx_dp.clone(); let j_scalars = &jump_scalars;
            let j_points = &jump_points; let fb_ref = Arc::clone(&fixed_base); 
            s.spawn(move || {
                worker_loop(idx % 2 == 0, j_scalars, j_points, target_projective, &fb_ref, state, tx_dp_clone, master_start, master_span, args.dp_bits, args.sub_bits, idx);
            });
        }
        drop(tx_dp);

        let mut last_ui = Instant::now();
        let mut last_total_hops_128: u128 = 0;
        let mut ui_dps = durable_records_count;
        let mut ram_seq = global_seq;
        let mut last_local_sync = Instant::now();
        let mut last_drive_sync = Instant::now();
        let mut current_gen = active_gen;
        let mut pending_checkpoint: Option<u64> = None;

        while KEEP_RUNNING.load(Ordering::Relaxed) {
            match rx_dp.recv_timeout(Duration::from_millis(100)) {
                Ok(msg) => {
                    let mut buf = [0u8; RECORD_SIZE];
                    buf[0..8].copy_from_slice(&ram_seq.to_be_bytes()); ram_seq += 1;

                    let limbs = [msg.key.x_3, msg.key.x_2, msg.key.x_1, msg.key.x_0];
                    for i in 0..4 {
                        buf[8 + i*8 .. 16 + i*8].copy_from_slice(&limbs[i].to_be_bytes());
                        buf[41 + i*8 .. 49 + i*8].copy_from_slice(&msg.scalar_limbs[3-i].to_be_bytes());
                    }
                    
                    buf[40] = msg.target_coeff as u8;

                    let mut hasher = Hasher::new(); hasher.update(&buf[0..73]);
                    buf[73..77].copy_from_slice(&hasher.finalize().to_be_bytes());

                    if let Some(existing_paths) = dp_map.get(&msg.key) {
                        let c1 = msg.target_coeff;
                        let s1 = limbs_to_fr(msg.scalar_limbs);
                        
                        for &(c2, ext_scalar) in existing_paths {
                            if c1 != c2 {
                                let s2 = limbs_to_fr(ext_scalar);
                                let s_diff = s2 - s1;
                                let c_diff = c1 as i32 - c2 as i32;
                                
                                let c_diff_fr = if c_diff > 0 { Fr::from(c_diff as u64) } else { -Fr::from((-c_diff) as u64) };
                                let final_scalar = s_diff * c_diff_fr.inverse().unwrap(); 
                                
                                verify_and_save(final_scalar, &target_bytes, &fixed_base);
                            }
                        }
                    }
                    dp_map.entry(msg.key).or_default().push((msg.target_coeff, msg.scalar_limbs));
                    ui_dps += 1;
                    
                    if tx_disk.send(DiskMsg::Record(buf)).is_err() {
                        eprintln!("🔥 Lỗi: Luồng Ghi Đĩa đã ngắt kết nối.");
                        KEEP_RUNNING.store(false, Ordering::Release); break;
                    }
                },
                Err(crossbeam_channel::RecvTimeoutError::Timeout) => { },
                Err(crossbeam_channel::RecvTimeoutError::Disconnected) => { break; }
            }

            let now = Instant::now();
            if now.duration_since(last_local_sync).as_secs() >= 60 {
                if tx_disk.send(DiskMsg::LocalFlushReq).is_err() { KEEP_RUNNING.store(false, Ordering::Release); break; }
                last_local_sync = now;
            }

            if pending_checkpoint.is_none() && now.duration_since(last_drive_sync).as_secs() >= 300 {
                if tx_disk.send(DiskMsg::CheckpointBarrier { gen: current_gen, is_final: false }).is_err() { KEEP_RUNNING.store(false, Ordering::Release); break; }
                pending_checkpoint = Some(current_gen);
            }

            if let Some(expected_gen) = pending_checkpoint {
                match rx_ack.try_recv() {
                    Ok((ack_gen, success)) => {
                        if ack_gen == expected_gen && success { current_gen += 1; last_drive_sync = Instant::now(); } 
                        else { last_drive_sync = Instant::now() - Duration::from_secs(240); }
                        pending_checkpoint = None; 
                    },
                    Err(crossbeam_channel::TryRecvError::Empty) => { },
                    Err(crossbeam_channel::TryRecvError::Disconnected) => { KEEP_RUNNING.store(false, Ordering::Release); break; }
                }
            }

            if now.duration_since(last_ui).as_secs() >= 30 {
                let mut epoch_total_hops_128: u128 = 0;
                for st in &worker_states { epoch_total_hops_128 += st.keys_scanned.load(Ordering::Relaxed) as u128; }
                let elapsed = now.duration_since(last_ui).as_secs_f64();
                let delta = epoch_total_hops_128.saturating_sub(last_total_hops_128);
                let rate = if elapsed > 0.0 { (delta as f64 / elapsed) as u128 } else { 0 };
                last_ui = now; last_total_hops_128 = epoch_total_hops_128;

                println!("➤ Hops: {} | DPs (RAM): {} | Speed: {} Hops/s",
                    compact_u128(epoch_total_hops_128), compact_u128(ui_dps as u128), compact_u128(rate));
                std::io::stdout().flush().unwrap();
            }
        }

        if let Some(expected_gen) = pending_checkpoint {
            println!("\n[*] Đang đợi Checkpoint {} định kỳ hoàn tất trước khi Shutdown...", expected_gen);
            if let Ok((ack_gen, success)) = rx_ack.recv_timeout(Duration::from_secs(120)) {
                if ack_gen == expected_gen && success { current_gen += 1; }
            }
        }

        println!("\n[!] Tín hiệu Shutdown. Đang thiết lập Final Checkpoint trước khi tắt...");
        if tx_disk.send(DiskMsg::CheckpointBarrier { gen: current_gen, is_final: true }).is_ok() {
            if let Ok((ack_gen, success)) = rx_ack.recv_timeout(Duration::from_secs(120)) {
                if ack_gen == current_gen && success { println!("[+] Final Checkpoint {} hoàn tất. Dữ liệu ĐÃ AN TOÀN.", current_gen); } 
                else { eprintln!("[-] Lỗi Ghi đĩa ở Final Checkpoint. Vui lòng kiểm tra Ổ cứng/Drive."); }
            }
        }
    });
    
    drop(tx_disk); let _ = disk_handle.join();
}
