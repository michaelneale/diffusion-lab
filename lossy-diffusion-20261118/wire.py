"""Actual CPU byte codec for local activation-transport perturbation, batch1 only."""
import struct
import numpy as np
HEADER=struct.Struct('<4sBBHII')
GROUP=64

def encode(x,bits):
    assert x.ndim==3 and x.shape[0]==1 and x.shape[-1]%GROUP==0
    assert bits in (4,8,16,17) and np.isfinite(x).all()
    rows,width=x.shape[1:]
    header=HEADER.pack(b'ACT1',1,bits,GROUP,rows,width)
    if bits==17:
        values=(x.astype('<f4').view('<u4')>>16).astype('<u2')
        restored=(values.astype('<u4')<<16).view('<f4')
        assert np.array_equal(restored,x), 'BF16 lossless input must already be BF16'
        return header+values.tobytes()
    if bits==16:
        values=x.astype('<f2')
        assert np.isfinite(values).all()
        return header+values.tobytes()
    grouped=x.astype(np.float32).reshape(-1,GROUP)
    qmax=(1<<(bits-1))-1
    scales=np.maximum(np.max(np.abs(grouped),axis=-1)/qmax,np.float32(2**-24)).astype('<f2')
    assert np.isfinite(scales).all() and (scales>0).all()
    q=np.clip(np.rint(grouped/scales.astype(np.float32)[:,None]),-qmax,qmax).astype(np.int16)
    codes=(q+qmax).astype(np.uint8).reshape(-1)
    if bits==4:codes=codes[::2] | (codes[1::2]<<4)
    return header+scales.tobytes()+codes.tobytes()

def decode(packet):
    magic,version,bits,group,rows,width=HEADER.unpack_from(packet)
    assert magic==b'ACT1' and version==1 and bits in (4,8,16,17) and group==GROUP
    assert rows>0 and width>0 and width%group==0
    n=rows*width;start=HEADER.size
    expected=start+n*(16 if bits==17 else bits)//8+(0 if bits in (16,17) else n//group*2)
    assert len(packet)==expected
    if bits==17:return (np.frombuffer(packet,dtype='<u2',offset=start).astype('<u4')<<16).view('<f4').reshape(1,rows,width)
    if bits==16:return np.frombuffer(packet,dtype='<f2',offset=start).astype(np.float32).reshape(1,rows,width)
    scales=np.frombuffer(packet,dtype='<f2',offset=start,count=n//group).astype(np.float32)
    assert np.isfinite(scales).all() and (scales>0).all()
    codes=np.frombuffer(packet,dtype=np.uint8,offset=start+2*len(scales))
    if bits==4:
        unpacked=np.empty(n,dtype=np.uint8);unpacked[::2]=codes&15;unpacked[1::2]=codes>>4;codes=unpacked
    qmax=(1<<(bits-1))-1
    assert (codes<=2*qmax).all()
    q=(codes.astype(np.int16)-qmax).reshape(-1,group)
    return (q*scales[:,None]).reshape(1,rows,width)

def self_test():
    rng=np.random.default_rng(42)
    for bits in (4,8,16):
        for x in (np.zeros((1,2,128),np.float32),rng.normal(size=(1,3,2816)).astype(np.float32),np.full((1,1,64),1e-9,np.float32),np.linspace(-100,100,64,dtype=np.float32).reshape(1,1,64)):
            packet=encode(x,bits);out=decode(packet)
            assert out.shape==x.shape and np.isfinite(out).all()
            assert len(packet)==16+x.size*bits//8+(0 if bits==16 else x.size//64*2)
            if bits==16:assert np.array_equal(out,x.astype(np.float16).astype(np.float32))
            else:
                scale=np.maximum(np.max(np.abs(x.reshape(-1,64)),axis=-1)/((1<<(bits-1))-1),2**-24).astype(np.float16).astype(np.float32)
                assert np.all(np.max(np.abs((out-x).reshape(-1,64)),axis=-1)<=scale*.502+1e-6)
            try:decode(packet[:-1])
            except (AssertionError,ValueError):pass
            else:raise AssertionError('accepted truncated packet')

    x=(rng.normal(size=(1,3,2816)).astype('<f4').view('<u4') & np.uint32(0xffff0000)).view('<f4')
    assert np.array_equal(decode(encode(x,17)),x)
    assert len(encode(x,17))==16+2*x.size

if __name__=='__main__':
    self_test();print('codec self-tests passed')
