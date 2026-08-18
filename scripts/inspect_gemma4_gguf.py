#!/usr/bin/env python3
"""Minimal GGUF v3 reader: dumps config KV + tensor inventory.
Used to settle Gemma 4 questions (Findings 4 & 5)."""
import struct, sys, collections

path = sys.argv[1] if len(sys.argv) > 1 else '/home/xander/Downloads/models/gemma-4-12b-it-Q4_K_M.gguf'
f = open(path, 'rb')
assert f.read(4) == b'GGUF'
struct.unpack('<I', f.read(4))[0]  # version
nt = struct.unpack('<Q', f.read(8))[0]
nk = struct.unpack('<Q', f.read(8))[0]
print(f'tensors={nt} kv={nk}')

def rstr():
    n = struct.unpack('<Q', f.read(8))[0]
    return f.read(n).decode('utf-8', 'replace')

FMT = {0:'<B',1:'<b',2:'<H',3:'<h',4:'<I',5:'<i',6:'<f',7:'<?',10:'<Q',11:'<q',12:'<d'}
def rval(t):
    if t == 8: return rstr()
    if t == 9:
        et = struct.unpack('<I', f.read(4))[0]
        n = struct.unpack('<Q', f.read(8))[0]
        return [rval(et) for _ in range(n)]
    if t in FMT:
        sz = struct.calcsize(FMT[t])
        return struct.unpack(FMT[t], f.read(sz))[0]
    raise Exception(f'bad type {t}')

kv = {}
for _ in range(nk):
    k = rstr(); t = struct.unpack('<I', f.read(4))[0]
    kv[k] = rval(t)

print('\n=== general + gemma4 config ===')
for k in sorted(kv):
    if k.startswith('gemma4.') or k == 'general.architecture' or k.startswith('tokenizer.'):
        v = kv[k]
        if isinstance(v, list) and len(v) > 20:
            v = f'<list len {len(v)}>'
        print(f'  {k} = {v}')

tinfos = []
for _ in range(nt):
    name = rstr()
    ndim = struct.unpack('<I', f.read(4))[0]
    dims = [struct.unpack('<Q', f.read(8))[0] for _ in range(ndim)]
    ttype = struct.unpack('<I', f.read(4))[0]
    off = struct.unpack('<Q', f.read(8))[0]
    tinfos.append((name, ndim, dims, ttype, off))

suf = collections.Counter()
for name, nd, d, tt, off in tinfos:
    parts = name.split('.')
    if parts[0] == 'blk' and len(parts) >= 3:
        suf['.'.join(parts[2:])] += 1

print('\n=== distinct blk.N.<suffix> (count) ===')
for s, c in sorted(suf.items()):
    print(f'  {c:4d}  {s}')

print('\n=== blk.0.* (full) ===')
for name, nd, d, tt, off in tinfos:
    if name.startswith('blk.0.'):
        print(f'  {name}  dims={d}  ttype={tt}')

print('\n=== top-level (non-blk) tensors ===')
for name, nd, d, tt, off in tinfos:
    if not name.startswith('blk.'):
        print(f'  {name}  dims={d}  ttype={tt}')
