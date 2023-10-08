#[macro_use]
extern crate criterion;
extern crate rand_core;
extern crate group;
use ff::Field;
use criterion::{BenchmarkId, Criterion};

use halo2_proofs::arithmetic::parallelize;
use halo2_proofs::halo2curves::pasta::vesta;
use rand_core::OsRng;

pub fn bench_ipa_mul(c: &mut Criterion) {
    let mut group = c.benchmark_group("ipa_mul");
    for k in 15..28 {
        let extended_len = 1 << k;
        let mut rand_ext_vec: Vec<vesta::Scalar> = (0..extended_len)
            .map(|_| vesta::Scalar::random(OsRng))
            .collect();
        group.bench_function(BenchmarkId::new("k", k), |b| {
            b.iter(|| {
                let rand_ele = vesta::Scalar::random(&mut OsRng);
                parallelize(&mut rand_ext_vec, |rand_ext_vec, _| {
                    for value in rand_ext_vec.iter_mut() {
                        let _ = *value * rand_ele;
                    }
                })
            });
        });
    }
}

criterion_group!{
    name=benches;
    config=Criterion::default(); 
    targets=bench_ipa_mul
}
criterion_main!(benches);