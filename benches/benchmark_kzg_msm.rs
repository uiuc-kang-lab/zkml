#[macro_use]
extern crate criterion;
extern crate rand_core;
extern crate group;
use ff::Field;
use group::prime::PrimeCurveAffine;
use criterion::{BenchmarkId, Criterion};

use halo2_proofs::{
    halo2curves::bn256::G1Affine,
    arithmetic::best_multiexp,
};
use rand_core::OsRng;

pub fn bench_kzg_multiexp(c: &mut Criterion) {
    let mut group = c.benchmark_group("kzg_msm");
    
    for k in 17..18 {
        let size = 1 << k;
        let mut rng = OsRng;

        let multiexp_scalars: Vec<<G1Affine as PrimeCurveAffine>::Scalar> = (0..size)
                .map(|_| <G1Affine as PrimeCurveAffine>::Scalar::random(&mut rng))
                .collect();
        let multiexp_bases: Vec<G1Affine> = (0..size)
                .map(|_| G1Affine::random(&mut rng))
                .collect();
        
        group.bench_function(BenchmarkId::new("k", k), |b| {
            b.iter(|| {
                best_multiexp(&multiexp_scalars, &multiexp_bases);
                criterion::black_box(&multiexp_scalars);
                criterion::black_box(&multiexp_bases);
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