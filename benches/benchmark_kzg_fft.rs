#[macro_use]
extern crate criterion;
extern crate rand_core;
extern crate group;
use group::ff::Field;
use criterion::{BenchmarkId, Criterion};

use halo2_proofs::{
  halo2curves::bn256::Fr,
  arithmetic::best_fft,
};
use rand_core::OsRng;

pub fn bench_kzg_fft(c: &mut Criterion) {
  let mut group = c.benchmark_group("kzg_fft");
  for k in 17..18 {
    group.bench_function(BenchmarkId::new("k", k), |b| {
      let mut a = (0..(1 << k)).map(|_| Fr::random(OsRng)).collect::<Vec<_>>();
      let omega = Fr::random(OsRng);
      b.iter(|| {
        best_fft(&mut a, omega, k as u32);
        criterion::black_box(&a);
      });
    });
  }
}

criterion_group!{
  name=benches; 
  config=Criterion::default().sample_size(10); 
  targets=bench_kzg_fft
}
criterion_main!(benches);