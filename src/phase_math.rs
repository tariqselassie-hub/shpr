use std::f64::consts::PI;

/// Fast deterministic splitmix64 PRNG. Seeded once, then polled for each dimension.
#[inline]
pub fn next_splitmix64(state: &mut u64) -> u64 {
    let mut z = *state;
    z = z.wrapping_add(0x9E3779B97F4A7C15);
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D049BB133111EB);
    let out = z ^ (z >> 31);
    *state = z;
    out
}

/// Mix an integer seed and string key into a stable u64 starting state for splitmix64.
#[inline]
pub fn mix_str_key_into_state(seed: u64, key: &str) -> u64 {
    let mut h = seed.wrapping_add(0x9E3779B97F4A7C15);
    for b in key.bytes() {
        h = h.wrapping_add(b as u64);
        h = (h ^ (h >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
        h = (h ^ (h >> 27)).wrapping_mul(0x94D049BB133111EB);
    }
    h ^ (h >> 31)
}

/// Normalizes any phase angle in radians to the principal interval [-pi, pi].
#[inline]
pub fn normalize_angle(angle: f64) -> f64 {
    let mut a = (angle + PI) % (2.0 * PI);
    if a < 0.0 {
        a += 2.0 * PI;
    }
    a - PI
}
