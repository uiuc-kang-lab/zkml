#[macro_use]
extern crate criterion;
extern crate rand_core;
extern crate group;
use ff::Field;
use criterion::{BenchmarkId, Criterion};
use halo2_proofs::poly::commitment::ParamsProver;
use halo2_proofs::{
    poly::kzg::commitment::ParamsKZG,
    halo2curves::bn256::{Fr, Bn256},
    arithmetic::best_multiexp,
};
use rand_core::OsRng;

pub fn bench_kzg_multiexp(c: &mut Criterion) {
    let mut group = c.benchmark_group("kzg_msm");
    
    for k in 13..20 {
        let size = 1 << k;
        let mut rng = OsRng;

        let multiexp_scalars: Vec<Fr> = (0..size)
                .map(|_| Fr::random(&mut rng))
                .collect();
        let params = ParamsKZG::<Bn256>::new(k);
        let multiexp_bases = params.get_g();
        
        group.bench_function(BenchmarkId::new("k", k), |b| {
            b.iter(|| {
                best_multiexp(&multiexp_scalars, &multiexp_bases);
            });
        });
    }
}

criterion_group!{
    name=benches; 
    config=Criterion::default().sample_size(10); 
    targets=bench_kzg_multiexp
  }
criterion_main!(benches);