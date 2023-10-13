#[macro_use]
extern crate criterion;
extern crate rand_core;
extern crate group;
use ff::{Field, PrimeField};
use criterion::{BenchmarkId, Criterion};

use halo2_proofs::halo2curves::pasta::vesta;
use halo2_proofs::plonk::Error;
use halo2_proofs::arithmetic::CurveAffine;
use rand_core::OsRng;
use std::{
    collections::HashMap,
    hash::{Hash, Hasher},
};

fn simulate_permute_expression_pair(k: i32) -> Result<vesta::Scalar, Error> {
    let params_n = 1<<k;
    let usable_rows = params_n as usize;
    let mut rng = OsRng;

    let mut permuted_input_expression: Vec<vesta::Scalar> = (0..params_n)
    .map(|_| vesta::Scalar::random(&mut rng))
    .collect();
    permuted_input_expression.truncate(usable_rows);

    // Sort input lookup expression values
    permuted_input_expression.sort();

    struct ScalarWrapper<C: CurveAffine> {
        value: C::ScalarExt,
    }
    
    impl<C: CurveAffine> ScalarWrapper<C> {
        fn new(value: C::ScalarExt) -> Self {
            ScalarWrapper { value }
        }
    }
    
    impl<C: CurveAffine> PartialEq for ScalarWrapper<C> {
        fn eq(&self, other: &Self) -> bool {
            // Implement PartialEq based on your requirements
            self.value == other.value
        }
    }
    
    impl<C: CurveAffine> Eq for ScalarWrapper<C> {}
    
    impl<C: CurveAffine> Hash for ScalarWrapper<C> {
        fn hash<H: Hasher>(&self, state: &mut H) {
            // Convert the value to a string and hash the string
            self.value.to_repr().as_ref().hash(state);
        }
    }

    // A BTreeMap of each unique element in the table expression and its count
    //let mut leftover_table_map: BTreeMap<vesta::Scalar, u32> = permuted_input_expression
    //    .iter()
    //    .take(usable_rows)
    //    .fold(BTreeMap::new(), |mut acc, coeff| {
    //        *acc.entry(*coeff).or_insert(0) += 1;
    //        acc
    //    });

    let mut leftover_table_map: HashMap<ScalarWrapper<vesta::Affine>, u32> = HashMap::new();
    for &coeff in permuted_input_expression.iter().take(usable_rows) {
        let wrapper = ScalarWrapper::new(coeff);
        *leftover_table_map.entry(wrapper).or_insert(0) += 1;
    }
    
    let mut permuted_table_coeffs = vec![vesta::Scalar::ZERO; usable_rows];

    let mut repeated_input_rows = permuted_input_expression
        .iter()
        .zip(permuted_table_coeffs.iter_mut())
        .enumerate()
        .filter_map(|(row, (input_value, table_value))| {
            // If this is the first occurrence of `input_value` in the input expression
            if row == 0 || *input_value != permuted_input_expression[row - 1] {
                *table_value = *input_value;
                // Remove one instance of input_value from leftover_table_map
                if let Some(count) = leftover_table_map.get_mut(&ScalarWrapper::new(*input_value)) {
                    assert!(*count > 0);
                    *count -= 1;
                    None
                } else {
                    // Return error if input_value not found
                    Some(Err(Error::ConstraintSystemFailure))
                }
            // If input value is repeated
            } else {
                Some(Ok(row))
            }
        })
        .collect::<Result<Vec<_>, _>>()?;

    // Populate permuted table at unfilled rows with leftover table elements
    for (coeff, count) in leftover_table_map.iter() {
        for _ in 0..*count {
            permuted_table_coeffs[repeated_input_rows.pop().unwrap()] = coeff.value;
        }
    }
    Ok(vesta::Scalar::one() + &permuted_input_expression[0])
}

pub fn bench_ipa_permute_expression_pair(c: &mut Criterion) {
    let mut group = c.benchmark_group("ipa_permute");
    
    for k in 13..20 {
        group.bench_function(BenchmarkId::new("k", k), |b| {
            b.iter(|| {
                let _ = simulate_permute_expression_pair(k);
            });
        });
    }
}

criterion_group!{
    name=benches; 
    config=Criterion::default().sample_size(10); 
    targets=bench_ipa_permute_expression_pair
}
criterion_main!(benches);