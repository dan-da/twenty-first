use arbitrary::Arbitrary;
use itertools::Itertools;
use proptest::prelude::*;
use proptest_arbitrary_interop::arb;
use test_strategy::proptest;

use twenty_first::shared_math::b_field_element::BFieldElement;
use twenty_first::shared_math::bfield_codec::BFieldCodec;
use twenty_first::shared_math::digest::Digest;
use twenty_first::shared_math::x_field_element::XFieldElement;

#[derive(Debug, Clone, PartialEq, Eq, BFieldCodec, Arbitrary)]
struct BFieldCodecTestStructA {
    a: u32,
    b: BFieldElement,
}

#[proptest]
fn integration_test_struct_a(#[strategy(arb())] test_struct: BFieldCodecTestStructA) {
    let encoding = test_struct.encode();
    let decoding = *BFieldCodecTestStructA::decode(&encoding).unwrap();
    prop_assert_eq!(test_struct, decoding);
}

#[derive(Debug, Clone, PartialEq, Eq, BFieldCodec, Arbitrary)]
struct BFieldCodecTestStructB {
    a: XFieldElement,
    b: Vec<(u64, Digest)>,
}

#[proptest]
fn integration_test_struct_b(#[strategy(arb())] test_struct: BFieldCodecTestStructB) {
    let encoding = test_struct.encode();
    let decoding = *BFieldCodecTestStructB::decode(&encoding).unwrap();
    prop_assert_eq!(test_struct, decoding);
}

#[derive(Debug, Clone, PartialEq, Eq, BFieldCodec, Arbitrary)]
enum BFieldCodecTestEnumA {
    A,
    B,
    C,
}

#[proptest]
fn integration_test_enum_a(#[strategy(arb())] test_enum: BFieldCodecTestEnumA) {
    let encoding = test_enum.encode();
    let decoding = *BFieldCodecTestEnumA::decode(&encoding).unwrap();
    prop_assert_eq!(test_enum, decoding);
}

#[derive(Debug, Clone, PartialEq, Eq, BFieldCodec, Arbitrary)]
enum BFieldCodecTestEnumB {
    A(u32),
    B(XFieldElement),
    C(Vec<(u64, Digest)>),
}

#[proptest]
fn integration_test_enum_b(#[strategy(arb())] test_enum: BFieldCodecTestEnumB) {
    let encoding = test_enum.encode();
    let decoding = *BFieldCodecTestEnumB::decode(&encoding).unwrap();
    prop_assert_eq!(test_enum, decoding);
}
