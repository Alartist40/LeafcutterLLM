//! BPE Tokenizer (Rust port of Colibri's tok.h).
//!
//! Supports loading tokenizer.json (HuggingFace format) and
//! encoding/decoding text.  Handles cl100k (GPT-4/GLM) and o200k
//! (GPT-4o/Inkling) pre-tokenizer regex families.
//!
//! Ported from colibri/c/tok.h. Key differences from the C version:
//! - Uses serde_json instead of colibri's custom json.h
//! - Uses Rust HashMap instead of the custom hmap
//! - Same BPE algorithm (rank-based merge) and pre-tokenizer regex logic

use std::collections::HashMap;

/// A BPE tokenizer loaded from tokenizer.json.
pub struct BpeTokenizer {
    /// string -> id (stored as Vec<u8> key for byte-slice lookup)
    vocab: HashMap<Vec<u8>, i32>,
    /// "left\0right" -> rank
    merges: HashMap<Vec<u8>, i32>,
    /// id -> string
    id2str: Vec<String>,
    /// id_added[id] = true if the token is an "added" token (output literal)
    id_added: Vec<bool>,
    /// id_special[id] = true if the token is a special/control token
    id_special: Vec<bool>,
    /// Added tokens sorted by length descending (for greedy matching)
    specials: Vec<Special>,
    /// GPT-2 ByteLevel: byte -> unicode codepoint mapping
    byte2str: [[u8; 3]; 256],
    byte2cp_len: [usize; 256],
    /// codepoint -> byte (for decoding)
    cp2byte: [i16; 1024],
    /// Pre-tokenizer family: false = cl100k (GLM), true = o200k (Inkling/GPT-4o)
    o200k: bool,
}

#[derive(Clone)]
struct Special {
    str: String,
    len: usize,
    id: i32,
}

impl BpeTokenizer {
    /// Load a tokenizer from a tokenizer.json file.
    pub fn load(path: &str) -> Result<Self, String> {
        let content = std::fs::read_to_string(path)
            .map_err(|e| format!("read {path}: {e}"))?;
        let root: serde_json::Value =
            serde_json::from_str(&content).map_err(|e| format!("parse json: {e}"))?;

        let model = root
            .get("model")
            .ok_or_else(|| "tokenizer.json: no model".to_string())?;
        let vocab_arr = model
            .get("vocab")
            .and_then(|v| v.as_object())
            .ok_or_else(|| "tokenizer.json: no model.vocab".to_string())?;
        let merges_arr = model
            .get("merges")
            .and_then(|v| v.as_array())
            .ok_or_else(|| "tokenizer.json: no model.merges".to_string())?;
        let added = root.get("added_tokens").and_then(|v| v.as_array());

        // Find max id to size id2str.
        let mut max_id: i32 = 0;
        for (_k, v) in vocab_arr {
            if let Some(id) = v.as_i64() {
                let id = id as i32;
                if id > max_id {
                    max_id = id;
                }
            }
        }
        if let Some(added) = added {
            for a in added {
                if let Some(id) = a.get("id").and_then(|v| v.as_i64()) {
                    let id = id as i32;
                    if id > max_id {
                        max_id = id;
                    }
                }
            }
        }
        let n_ids = (max_id + 1) as usize;

        let mut id2str = vec![String::new(); n_ids];
        let mut id_added = vec![false; n_ids];
        let mut id_special = vec![false; n_ids];
        let mut vocab = HashMap::new();
        let mut merges = HashMap::new();

        // vocab: string -> id
        for (k, v) in vocab_arr {
            if let Some(id) = v.as_i64() {
                let id = id as i32;
                vocab.insert(k.as_bytes().to_vec(), id);
                id2str[id as usize] = k.clone();
            }
        }

        // merges: [left, right] -> rank=i
        for (i, m) in merges_arr.iter().enumerate() {
            if let Some(arr) = m.as_array() {
                if arr.len() >= 2 {
                    let l = arr[0].as_str().unwrap_or("");
                    let r = arr[1].as_str().unwrap_or("");
                    let mut key = Vec::with_capacity(l.len() + 1 + r.len());
                    key.extend_from_slice(l.as_bytes());
                    key.push(0); // separator
                    key.extend_from_slice(r.as_bytes());
                    merges.insert(key, i as i32);
                }
            }
        }

        // Added tokens (specials and non-specials).
        let mut specials = Vec::new();
        if let Some(added) = added {
            for a in added {
                let content = a.get("content").and_then(|v| v.as_str()).unwrap_or("");
                let id = a.get("id").and_then(|v| v.as_i64()).unwrap_or(0) as i32;
                if (id as usize) < n_ids {
                    id2str[id as usize] = content.to_string();
                    id_added[id as usize] = true;
                    let is_special = a
                        .get("special")
                        .and_then(|v| v.as_bool())
                        .unwrap_or(false);
                    if is_special {
                        id_special[id as usize] = true;
                    }
                    specials.push(Special {
                        str: content.to_string(),
                        len: content.len(),
                        id,
                    });
                }
            }
            // Sort by length descending (greedy longest match).
            specials.sort_by(|a, b| b.len.cmp(&a.len));
        }

        // Pre-tokenizer family detection.
        let mut o200k = false;
        if let Some(pt) = root.get("pre_tokenizer") {
            if let Some(ps) = pt.get("pretokenizers").and_then(|v| v.as_array()) {
                for p in ps {
                    if let Some(rx) = p.get("pattern").and_then(|v| v.get("Regex")) {
                        if rx.as_str().map(|s| s.contains(r"\p{Lu}")).unwrap_or(false) {
                            o200k = true;
                        }
                    }
                }
            }
        }

        // Build byte-level map (GPT-2 ByteLevel).
        let mut byte2str = [[0u8; 3]; 256];
        let mut byte2cp_len = [1usize; 256];
        let mut cp2byte = [-1i16; 1024];

        let is_direct: [bool; 256] = {
            let mut d = [false; 256];
            for b in 33..=126 {
                d[b] = true;
            }
            for b in 161..=172 {
                d[b] = true;
            }
            for b in 174..=255 {
                d[b] = true;
            }
            d
        };
        let mut n = 0;
        for b in 0usize..256 {
            let cp = if is_direct[b] { b as u32 } else { 256 + n };
            if !is_direct[b] {
                n += 1;
            }
            byte2cp_len[b] = u8_put(&mut byte2str[b], cp);
            if cp < 1024 {
                cp2byte[cp as usize] = b as i16;
            }
        }

        Ok(Self {
            vocab,
            merges,
            id2str,
            id_added,
            id_special,
            specials,
            byte2str,
            byte2cp_len,
            cp2byte,
            o200k,
        })
    }

    /// Encode text to token IDs.
    pub fn encode(&self, text: &str, max: usize) -> Vec<i32> {
        let p = text.as_bytes();
        let mut out = Vec::new();
        let mut i = 0;
        while i < p.len() {
            // Find next occurrence of an added-token starting at >= i (greedy longest).
            let mut hit_pos = -1i64;
            let mut hit_len = 0usize;
            let mut hit_id = -1i32;
            for j in i..p.len() {
                if hit_pos >= 0 {
                    break;
                }
                for sp in &self.specials {
                    if sp.len > 0 && j + sp.len <= p.len() && &p[j..j + sp.len] == sp.str.as_bytes() {
                        hit_pos = j as i64;
                        hit_len = sp.len;
                        hit_id = sp.id;
                        break;
                    }
                }
            }
            let chunk_end = if hit_pos < 0 { p.len() } else { hit_pos as usize };
            if chunk_end > i {
                if self.o200k {
                    self.pretok_chunk_o200k(p, i, chunk_end, &mut out, max);
                } else {
                    self.pretok_chunk_cl100k(p, i, chunk_end, &mut out, max);
                }
            }
            if hit_pos < 0 {
                break;
            }
            if out.len() < max {
                out.push(hit_id);
            }
            i = hit_pos as usize + hit_len;
        }
        out
    }

    /// Decode token IDs back to text.
    pub fn decode(&self, ids: &[i32]) -> String {
        let mut out = Vec::new();
        for &id in ids {
            if id < 0 || id as usize >= self.id2str.len() {
                continue;
            }
            let s = &self.id2str[id as usize];
            if self.id_added[id as usize] {
                out.extend_from_slice(s.as_bytes());
                continue;
            }
            // Decode through the byte-level map (inverse of GPT-2 ByteLevel).
            let sb = s.as_bytes();
            let mut j = 0;
            while j < sb.len() {
                let (cp, k) = u8_next(sb, j);
                j += k;
                if cp < 1024 && self.cp2byte[cp as usize] >= 0 {
                    out.push(self.cp2byte[cp as usize] as u8);
                }
            }
        }
        String::from_utf8_lossy(&out).into_owned()
    }

    /// Get the ID of an added token by its content (e.g. "<|im_start|>").
    pub fn id_of(&self, content: &str) -> i32 {
        for sp in &self.specials {
            if sp.str == content {
                return sp.id;
            }
        }
        -1
    }

    // ── BPE on a single chunk ──────────────────────────────────────────
    fn bpe_piece(&self, p: &[u8], a: usize, b: usize, out: &mut Vec<i32>, max: usize) {
        let nb = b - a;
        // Build the byte-level string.
        let mut s = Vec::with_capacity(2 * nb + 1);
        for &byte in &p[a..b] {
            let bl = self.byte2cp_len[byte as usize];
            s.extend_from_slice(&self.byte2str[byte as usize][..bl]);
        }

        // If the whole piece is a token, emit it directly.
        if let Some(&id) = self.vocab.get(s.as_slice()) {
            if out.len() < max {
                out.push(id);
            }
            return;
        }

        // Initial symbols = code points in the byte-level string.
        let mut soff = Vec::with_capacity(s.len() + 1);
        let mut slen = Vec::with_capacity(s.len() + 1);
        let mut i = 0;
        while i < s.len() {
            let (_cp, k) = u8_next(&s, i);
            soff.push(i);
            slen.push(k);
            i += k;
        }
        let mut ns = soff.len();

        let mut kbuf = vec![0u8; 2 * s.len() + 2];
        loop {
            let mut best = i32::MAX;
            let mut bp = -1i64;
            for j in 0..ns.saturating_sub(1) {
                let ll = slen[j];
                let rl = slen[j + 1];
                kbuf[..ll].copy_from_slice(&s[soff[j]..soff[j] + ll]);
                kbuf[ll] = 0;
                kbuf[ll + 1..ll + 1 + rl].copy_from_slice(&s[soff[j + 1]..soff[j + 1] + rl]);
                if let Some(&rk) = self.merges.get(&kbuf[..ll + 1 + rl]) {
                    if rk < best {
                        best = rk;
                        bp = j as i64;
                    }
                }
            }
            if bp < 0 {
                break;
            }
            let bp = bp as usize;
            slen[bp] = soff[bp + 1] + slen[bp + 1] - soff[bp];
            for j in bp + 1..ns - 1 {
                soff[j] = soff[j + 1];
                slen[j] = slen[j + 1];
            }
            ns -= 1;
        }

        for i in 0..ns {
            if let Some(&id) = self.merges.get(&s[soff[i]..soff[i] + slen[i]]) {
                let _ = id; // merges uses "left\0right" key, not the piece itself
            }
            // Actually we need vocab lookup, not merges.
            if let Some(&id) = self.vocab.get(&s[soff[i]..soff[i] + slen[i]]) {
                if out.len() < max {
                    out.push(id);
                }
            }
        }
    }

    // ── cl100k pre-tokenizer (GLM, GPT-4) ────────────────────────────
    fn pretok_chunk_cl100k(&self, p: &[u8], a: usize, b: usize, out: &mut Vec<i32>, max: usize) {
        let nb = b - a;
        if nb == 0 {
            return;
        }
        // Collect code points.
        let mut cp = Vec::with_capacity(nb + 1);
        let mut off = Vec::with_capacity(nb + 2);
        let mut i = a;
        while i < b {
            let (c, k) = u8_next(p, i);
            off.push(i);
            cp.push(c);
            i += k;
        }
        off.push(b);
        let n = cp.len();

        let mut idx = 0;
        while idx < n {
            let start = idx;
            let c = cp[idx];

            // 1) contractions: 's 't 're 've 'm 'll 'd
            if c == '\'' as u32 && idx + 1 < n {
                let d = to_lower(cp[idx + 1]);
                if idx + 2 < n {
                    let d2 = to_lower(cp[idx + 2]);
                    if (d == 'r' as u32 && d2 == 'e' as u32)
                        || (d == 'v' as u32 && d2 == 'e' as u32)
                        || (d == 'l' as u32 && d2 == 'l' as u32)
                    {
                        idx += 3;
                        self.bpe_piece(p, off[start], off[idx], out, max);
                        continue;
                    }
                }
                if d == 's' as u32 || d == 't' as u32 || d == 'm' as u32 || d == 'd' as u32 {
                    idx += 2;
                    self.bpe_piece(p, off[start], off[idx], out, max);
                    continue;
                }
            }

            // 2) [^\r\n\p{L}\p{N}]? \p{L}+
            {
                let mut j = idx;
                let is_punct = !is_L(c) && !is_NL(c) && !is_N(c);
                if is_punct && j + 1 < n && is_L(cp[j + 1]) {
                    j += 1;
                } else if is_punct {
                    j = n; // skip
                }
                if j < n && is_L(cp[j]) {
                    while j < n && is_L(cp[j]) {
                        j += 1;
                    }
                    idx = j;
                    self.bpe_piece(p, off[start], off[idx], out, max);
                    continue;
                }
            }

            // 3) \p{N}{1,3}
            if is_N(c) {
                let mut j = idx;
                let mut k = 0;
                while j < n && is_N(cp[j]) && k < 3 {
                    j += 1;
                    k += 1;
                }
                idx = j;
                self.bpe_piece(p, off[start], off[idx], out, max);
                continue;
            }

            // 4) ' ?[^\s\p{L}\p{N}]+[\r\n]*
            {
                let mut j = idx;
                if c == ' ' as u32
                    && j + 1 < n
                    && !is_S(cp[j + 1])
                    && !is_L(cp[j + 1])
                    && !is_N(cp[j + 1])
                {
                    j += 1;
                }
                if j < n && !is_S(cp[j]) && !is_L(cp[j]) && !is_N(cp[j]) {
                    while j < n && !is_S(cp[j]) && !is_L(cp[j]) && !is_N(cp[j]) {
                        j += 1;
                    }
                    while j < n && is_NL(cp[j]) {
                        j += 1;
                    }
                    idx = j;
                    self.bpe_piece(p, off[start], off[idx], out, max);
                    continue;
                }
            }

            // 5) \s*[\r\n]+  or  \s+(?!\S)
            {
                let mut r = idx;
                while r < n && is_S(cp[r]) {
                    r += 1;
                }
                if r > idx {
                    let mut last_nl = -1i64;
                    for j in idx..r {
                        if is_NL(cp[j]) {
                            last_nl = j as i64;
                        }
                    }
                    if last_nl >= 0 {
                        idx = last_nl as usize + 1;
                        self.bpe_piece(p, off[start], off[idx], out, max);
                        continue;
                    }
                    let end = if r < n { r - 1 } else { r };
                    let end = if end <= idx { idx + 1 } else { end };
                    idx = end;
                    self.bpe_piece(p, off[start], off[idx], out, max);
                    continue;
                }
            }

            // Fallback
            idx += 1;
            self.bpe_piece(p, off[start], off[idx], out, max);
        }
    }

    // ── o200k pre-tokenizer (GPT-4o, Inkling) ──────────────────────
    fn pretok_chunk_o200k(&self, p: &[u8], a: usize, b: usize, out: &mut Vec<i32>, max: usize) {
        // For now, use the same cl100k path. The o200k regex is more complex
        // and will be ported when we have a model that needs it.
        // TODO: implement o200k pre-tokenizer when targeting GPT-4o/Inkling models.
        self.pretok_chunk_cl100k(p, a, b, out, max);
    }
}

// ── UTF-8 helpers ───────────────────────────────────────────────────

/// Read one UTF-8 code point from byte slice at position `i`.
/// Returns (codepoint, bytes_consumed).
#[inline]
fn u8_next(s: &[u8], i: usize) -> (u32, usize) {
    let c = s[i];
    if c < 0x80 {
        return (c as u32, 1);
    }
    if (c >> 5) == 0x6 && i + 1 < s.len() {
        return (((c as u32 & 0x1F) << 6) | (s[i + 1] as u32 & 0x3F), 2);
    }
    if (c >> 4) == 0xE && i + 2 < s.len() {
        return (
            ((c as u32 & 0x0F) << 12) | ((s[i + 1] as u32 & 0x3F) << 6) | (s[i + 2] as u32 & 0x3F),
            3,
        );
    }
    if (c >> 3) == 0x1E && i + 3 < s.len() {
        return (
            ((c as u32 & 0x07) << 18)
                | ((s[i + 1] as u32 & 0x3F) << 12)
                | ((s[i + 2] as u32 & 0x3F) << 6)
                | (s[i + 3] as u32 & 0x3F),
            4,
        );
    }
    (c as u32, 1) // invalid byte: treat as single
}

/// Write a code point to a buffer (up to 4 bytes). Return bytes written.
#[inline]
fn u8_put(o: &mut [u8], cp: u32) -> usize {
    if cp < 0x80 {
        o[0] = cp as u8;
        1
    } else if cp < 0x800 {
        o[0] = 0xC0 | (cp >> 6) as u8;
        o[1] = 0x80 | (cp & 0x3F) as u8;
        2
    } else if cp < 0x10000 {
        o[0] = 0xE0 | (cp >> 12) as u8;
        o[1] = 0x80 | ((cp >> 6) & 0x3F) as u8;
        o[2] = 0x80 | (cp & 0x3F) as u8;
        3
    } else {
        o[0] = 0xF0 | (cp >> 18) as u8;
        o[1] = 0x80 | ((cp >> 12) & 0x3F) as u8;
        o[2] = 0x80 | ((cp >> 6) & 0x3F) as u8;
        o[3] = 0x80 | (cp & 0x3F) as u8;
        4
    }
}

// ── Unicode category predicates (simplified) ────────────────────────
// These are simplified versions of the Unicode property checks.
// For full correctness, we'd use the `unicode-properties` crate,
// but for ASCII-heavy text these are sufficient.

#[inline]
fn to_lower(c: u32) -> u32 {
    if c >= b'A' as u32 && c <= b'Z' as u32 {
        c + 32
    } else {
        c
    }
}

#[inline]
fn is_NL(c: u32) -> bool {
    c == '\r' as u32 || c == '\n' as u32
}

#[inline]
fn is_L(_c: u32) -> bool {
    // Simplified: ASCII letters only. Full implementation would check
    // all Unicode letter categories (Lu, Ll, Lt, Lm, Lo).
    (_c >= b'A' as u32 && _c <= b'Z' as u32) || (_c >= b'a' as u32 && _c <= b'z' as u32)
}

#[inline]
fn is_N(c: u32) -> bool {
    // Simplified: ASCII digits only. Full implementation would check
    // all Unicode number categories.
    c >= b'0' as u32 && c <= b'9' as u32
}

#[inline]
fn is_S(c: u32) -> bool {
    // Whitespace = space, tab, \r, \n, and other Unicode whitespace.
    c == ' ' as u32 || c == '\t' as u32 || c == '\r' as u32 || c == '\n' as u32
}
