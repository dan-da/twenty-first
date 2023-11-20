use std::fmt::Debug;
use std::fmt::Display;
use std::marker::PhantomData;
use std::slice::Iter;

use anyhow::anyhow;
use anyhow::bail;
use anyhow::Result;
use itertools::Itertools;
use num_traits::One;
use num_traits::Zero;

// Re-export the derive macro so that it can be used in other crates without having to add
// an explicit dependency on `bfieldcodec_derive` to their Cargo.toml.
pub use bfieldcodec_derive::BFieldCodec;

use super::b_field_element::BFieldElement;

/// BFieldCodec
///
/// This trait provides functions for encoding to and decoding from a
/// Vec of BFieldElements. This encoding does not record the size of
/// objects nor their type information; this is
/// the responsibility of the decoder.
pub trait BFieldCodec {
    type Error: Into<Box<dyn std::error::Error + Send + Sync + 'static>> + Debug + Display;

    fn decode(sequence: &[BFieldElement]) -> Result<Box<Self>, Self::Error>;
    fn encode(&self) -> Vec<BFieldElement>;

    /// Returns the length in number of BFieldElements if it is known at compile-time.
    /// Otherwise, None.
    fn static_length() -> Option<usize>;
}

// The underlying type of a BFieldElement is a u64. A single u64 does not fit in one BFieldElement.
// Therefore, deriving the BFieldCodec for BFieldElement using the derive macro will result in a
// BFieldCodec implementation that encodes a single BFieldElement as two BFieldElements.
// This is not desired. Hence, BFieldCodec is implemented manually for BFieldElement.
impl BFieldCodec for BFieldElement {
    type Error = anyhow::Error;

    fn decode(sequence: &[BFieldElement]) -> Result<Box<Self>> {
        if sequence.len() != 1 {
            bail!("trying to decode more or less than one BFieldElements as one BFieldElement");
        }
        let element_zero = sequence[0];
        Ok(Box::new(element_zero))
    }

    fn encode(&self) -> Vec<BFieldElement> {
        [*self].to_vec()
    }

    fn static_length() -> Option<usize> {
        Some(1)
    }
}

impl BFieldCodec for u128 {
    type Error = anyhow::Error;

    fn decode(sequence: &[BFieldElement]) -> Result<Box<Self>> {
        if sequence.len() != 4 {
            bail!(
                "Cannot decode sequence of length {} =/= 4 as u128.",
                sequence.len()
            );
        }
        if !sequence.iter().all(|s| s.value() <= u32::MAX as u64) {
            bail!(
                "Could not parse sequence of BFieldElements {:?} as u128.",
                sequence
            );
        }
        return Ok(Box::new(
            sequence
                .iter()
                .enumerate()
                .map(|(i, s)| (s.value() as u128) << (i * 32))
                .sum(),
        ));
    }

    fn encode(&self) -> Vec<BFieldElement> {
        (0..4)
            .map(|i| (*self >> (i * 32)) as u64 & u32::MAX as u64)
            .map(BFieldElement::new)
            .collect_vec()
    }

    fn static_length() -> Option<usize> {
        Some(4)
    }
}

impl BFieldCodec for u64 {
    type Error = anyhow::Error;

    fn decode(sequence: &[BFieldElement]) -> Result<Box<Self>> {
        if sequence.len() != 2 {
            bail!(
                "Cannot decode sequence of length {} =/= 2 as u64.",
                sequence.len()
            );
        }
        if !sequence.iter().all(|s| s.value() <= u32::MAX as u64) {
            bail!(
                "Could not parse sequence of BFieldElements {:?} as u64.",
                sequence
            );
        }
        return Ok(Box::new(
            sequence
                .iter()
                .enumerate()
                .map(|(i, s)| s.value() << (i * 32))
                .sum(),
        ));
    }

    fn encode(&self) -> Vec<BFieldElement> {
        (0..2)
            .map(|i| (*self >> (i * 32)) & u32::MAX as u64)
            .map(BFieldElement::new)
            .collect_vec()
    }

    fn static_length() -> Option<usize> {
        Some(2)
    }
}

impl BFieldCodec for bool {
    type Error = anyhow::Error;

    fn decode(sequence: &[BFieldElement]) -> Result<Box<Self>> {
        if sequence.len() != 1 {
            bail!(
                "Cannot decode sequence of {} =/= 1 BFieldElements as bool.",
                sequence.len()
            );
        }
        Ok(Box::new(match sequence[0].value() {
            0 => false,
            1 => true,
            n => bail!("Failed to parse BFieldElement {n} as bool."),
        }))
    }

    fn encode(&self) -> Vec<BFieldElement> {
        vec![BFieldElement::new(*self as u64)]
    }

    fn static_length() -> Option<usize> {
        Some(1)
    }
}

impl BFieldCodec for u32 {
    type Error = anyhow::Error;

    fn decode(sequence: &[BFieldElement]) -> Result<Box<Self>> {
        if sequence.len() != 1 {
            bail!(
                "Cannot decode sequence of length {} =/= 1 BFieldElements as u32.",
                sequence.len()
            );
        }
        let value = sequence[0].value();
        if value > u32::MAX as u64 {
            bail!("Cannot decode BFieldElement {value} as u32.");
        }
        Ok(Box::new(value as u32))
    }

    fn encode(&self) -> Vec<BFieldElement> {
        vec![BFieldElement::new(*self as u64)]
    }

    fn static_length() -> Option<usize> {
        Some(1)
    }
}

impl<T: BFieldCodec, S: BFieldCodec> BFieldCodec for (T, S)
where
    T::Error: Into<anyhow::Error>,
    S::Error: Into<anyhow::Error>,
{
    type Error = anyhow::Error;

    fn decode(str: &[BFieldElement]) -> Result<Box<Self>> {
        let mut index = 0;
        // decode T
        let len_t = match T::static_length() {
            Some(len) => len,
            None => {
                let length = match str.get(index) {
                    Some(bfe) => bfe.value() as usize,
                    None => bail!(
                        "Prepended length of type T satisfying unsized BFieldCodec does not exist"
                    ),
                };
                index += 1;
                length
            }
        };

        if str.len() < index + len_t {
            bail!("Sequence too short to decode tuple, element 0");
        }
        let t = *T::decode(&str[index..index + len_t]).map_err(|e| e.into())?;
        index += len_t;

        // decode S
        let len_s = match S::static_length() {
            Some(len) => len,
            None => {
                let length = match str.get(index) {
                    Some(bfe) => bfe.value() as usize,
                    None => bail!(
                        "Prepended length of type S satisfying unsized BFieldCodec does not exist"
                    ),
                };
                index += 1;
                length
            }
        };

        if str.len() < index + len_s {
            bail!("Sequence too short to decode tuple, element 1");
        }
        let s = *S::decode(&str[index..index + len_s]).map_err(|e| e.into())?;
        index += len_s;

        if index != str.len() {
            bail!("Error decoding (T,S): length mismatch");
        }

        Ok(Box::new((t, s)))
    }

    fn encode(&self) -> Vec<BFieldElement> {
        let mut str = vec![];
        let mut encoding_of_t = self.0.encode();
        let mut encoding_of_s = self.1.encode();
        if T::static_length().is_none() {
            str.push(BFieldElement::new(encoding_of_t.len() as u64));
        }
        str.append(&mut encoding_of_t);
        if S::static_length().is_none() {
            str.push(BFieldElement::new(encoding_of_s.len() as u64));
        }
        str.append(&mut encoding_of_s);
        str
    }

    fn static_length() -> Option<usize> {
        match (T::static_length(), S::static_length()) {
            (Some(sl_t), Some(sl_s)) => Some(sl_t + sl_s),
            _ => None,
        }
    }
}

impl<T: BFieldCodec> BFieldCodec for Option<T>
where
    T::Error: Into<anyhow::Error>,
{
    type Error = anyhow::Error;

    fn decode(str: &[BFieldElement]) -> Result<Box<Self>> {
        let is_some = match str.get(0) {
            Some(e) => {
                if e.is_one() || e.is_zero() {
                    e.is_one()
                } else {
                    bail!("Invalid option indicator: {e}")
                }
            }
            None => bail!("Cannot decode Option of T: empty sequence"),
        };

        if is_some {
            Ok(Box::new(Some(*T::decode(&str[1..]).map_err(|e| e.into())?)))
        } else {
            Ok(Box::new(None))
        }
    }

    fn encode(&self) -> Vec<BFieldElement> {
        let mut str = vec![];
        match self {
            None => {
                str.push(BFieldElement::zero());
            }
            Some(t) => {
                str.push(BFieldElement::one());
                str.append(&mut t.encode());
            }
        }
        str
    }

    fn static_length() -> Option<usize> {
        None
    }
}

impl<T: BFieldCodec, const N: usize> BFieldCodec for [T; N]
where
    T::Error: Into<anyhow::Error>,
{
    type Error = anyhow::Error;

    fn decode(sequence: &[BFieldElement]) -> Result<Box<Self>> {
        if N > 0 && sequence.is_empty() {
            bail!("Cannot decode empty sequence into [T; {N}]");
        }

        let vec_t = bfield_codec_decode_list(N, sequence).map_err(|e| anyhow!(e))?;
        let array = match vec_t.try_into() {
            Ok(array) => array,
            Err(_) => bail!("Cannot convert Vec<T> into [T; {N}]."),
        };
        Ok(Box::new(array))
    }

    fn encode(&self) -> Vec<BFieldElement> {
        bfield_codec_encode_list(self.iter())
    }

    fn static_length() -> Option<usize> {
        T::static_length().map(|len| len * N)
    }
}

impl<T: BFieldCodec> BFieldCodec for Vec<T>
where
    T::Error: Into<anyhow::Error>,
{
    type Error = anyhow::Error;

    fn decode(sequence: &[BFieldElement]) -> Result<Box<Self>> {
        if sequence.is_empty() {
            bail!("Cannot decode empty sequence into Vec<T>");
        }

        let indicated_num_elements = sequence[0].value() as usize;
        let vec_t = bfield_codec_decode_list(indicated_num_elements, &sequence[1..])
            .map_err(|e| anyhow!(e))?;
        Ok(Box::new(vec_t))
    }

    fn encode(&self) -> Vec<BFieldElement> {
        let num_elements = (self.len() as u64).into();
        let mut encoding = vec![num_elements];
        let mut encoded_items = bfield_codec_encode_list(self.iter());
        encoding.append(&mut encoded_items);
        encoding
    }

    fn static_length() -> Option<usize> {
        None
    }
}

/// The core of the [`BFieldCodec`] decoding logic for `Vec<T>` and `[T; N]`.
/// Decoding the length-prepending must be handled by the caller (if necessary).
fn bfield_codec_decode_list<T: BFieldCodec>(
    indicated_num_items: usize,
    sequence: &[BFieldElement],
) -> Result<Vec<T>>
where
    T::Error: Into<anyhow::Error>,
{
    // Initializing the vector with the indicated capacity potentially allows a DOS.
    let mut vec_t = vec![];

    let sequence_len = sequence.len();
    if T::static_length().is_some() {
        let item_length = T::static_length().unwrap();
        let maybe_vector_size = indicated_num_items.checked_mul(item_length);
        let Some(vector_size) = maybe_vector_size else {
            bail!("Length indication too large: {indicated_num_items} * {item_length}");
        };

        if sequence_len != vector_size {
            bail!(
                "Indicator for number of items must match sequence length. \
                List claims to contain {indicated_num_items} items. \
                Item size is {item_length}. Sequence length is {sequence_len}.",
            );
        }
        let raw_item_iter = sequence.chunks_exact(item_length);
        if !raw_item_iter.remainder().is_empty() {
            bail!("Could not chunk sequence into equal parts of size {item_length}.");
        }
        if raw_item_iter.len() != indicated_num_items {
            bail!(
                "List contains wrong number of items. Expected {indicated_num_items}, found {}.",
                raw_item_iter.len()
            );
        }
        for raw_item in raw_item_iter {
            let item = *T::decode(raw_item).map_err(|e| e.into())?;
            vec_t.push(item);
        }
    } else {
        let mut sequence_index = 0;
        for item_idx in 1..=indicated_num_items {
            let item_length = match sequence.get(sequence_index) {
                Some(len) => len.value() as usize,
                None => bail!(
                    "Index count mismatch while decoding List of T. \
                    Attempted to decode item {item_idx} of {indicated_num_items}.",
                ),
            };
            sequence_index += 1;
            if sequence_len < sequence_index + item_length {
                bail!(
                    "Sequence too short to decode List of T: \
                    {sequence_len} < {sequence_index} + {item_length}. \
                    Attempted to decode item {item_idx} of {indicated_num_items}.",
                );
            }
            let item = *T::decode(&sequence[sequence_index..sequence_index + item_length])
                .map_err(|e| e.into())?;
            sequence_index += item_length;
            vec_t.push(item);
        }
        if sequence_len != sequence_index {
            bail!("List contains too many items: {sequence_len} != {sequence_index}.");
        }
    }
    Ok(vec_t)
}

/// The core of the [`BFieldCodec`] encoding logic for `Vec<T>` and `[T; N]`.
/// Encoding the length-prepending must be handled by the caller (if necessary).
fn bfield_codec_encode_list<T: BFieldCodec>(item_iter: Iter<T>) -> Vec<BFieldElement> {
    if T::static_length().is_some() {
        item_iter.flat_map(|elem| elem.encode()).collect()
    } else {
        let mut encoding = vec![];
        for elem in item_iter {
            let mut element_encoded = elem.encode();
            let element_len = (element_encoded.len() as u64).into();
            encoding.push(element_len);
            encoding.append(&mut element_encoded);
        }
        encoding
    }
}

impl<T> BFieldCodec for PhantomData<T> {
    type Error = anyhow::Error;

    fn decode(sequence: &[BFieldElement]) -> Result<Box<Self>> {
        if !sequence.is_empty() {
            bail!(
                "Cannot decode non-empty BFE slice as phantom data; sequence length: {}",
                sequence.len()
            )
        }
        Ok(Box::new(PhantomData))
    }

    fn encode(&self) -> Vec<BFieldElement> {
        vec![]
    }

    fn static_length() -> Option<usize> {
        Some(0)
    }
}

#[cfg(test)]
mod tests {
    use proptest::collection::size_range;
    use proptest::collection::vec;
    use proptest::prelude::*;
    use proptest_arbitrary_interop::arb;
    use test_strategy::proptest;

    use crate::shared_math::{digest::Digest, tip5::Tip5, x_field_element::XFieldElement};

    use super::*;

    #[derive(Debug, PartialEq, Eq, test_strategy::Arbitrary)]
    struct BFieldCodecPropertyTestData<T>
    where
        T: 'static + BFieldCodec + Eq + Debug + Clone + for<'a> arbitrary::Arbitrary<'a>,
    {
        #[strategy(arb())]
        value: T,

        #[strategy(Just(#value.encode()))]
        encoding: Vec<BFieldElement>,

        #[strategy(vec(arb(), #encoding.len()))]
        #[filter(#encoding.iter().zip(#random_encoding.iter()).all(|(a, b)| a != b))]
        random_encoding: Vec<BFieldElement>,

        #[any(size_range(1..128).lift())]
        encoding_lengthener: Vec<BFieldElement>,

        #[strategy(0..=#encoding.len())]
        length_of_too_short_sequence: usize,
    }

    impl<T> BFieldCodecPropertyTestData<T>
    where
        T: 'static + BFieldCodec + Eq + Debug + Clone + for<'a> arbitrary::Arbitrary<'a>,
    {
        fn assert_bfield_codec_properties(&self) -> Result<(), TestCaseError> {
            self.assert_decoded_encoding_is_self()?;
            self.assert_decoding_too_long_encoding_fails()?;
            self.assert_decoding_too_short_encoding_fails()?;
            self.modify_each_element_and_assert_decoding_failure()?;
            self.assert_decoding_random_too_short_encoding_fails_gracefully()
        }

        fn assert_decoded_encoding_is_self(&self) -> Result<(), TestCaseError> {
            let encoding = self.encoding.to_owned();
            let decoding = T::decode(&encoding);
            let decoding = *decoding.unwrap();
            prop_assert_eq!(&self.value, &decoding);
            Ok(())
        }

        fn assert_decoding_too_long_encoding_fails(&self) -> Result<(), TestCaseError> {
            let too_long_encoding = [
                self.encoding.to_owned(),
                self.encoding_lengthener.to_owned(),
            ]
            .concat();
            prop_assert!(T::decode(&too_long_encoding).is_err());
            Ok(())
        }

        fn assert_decoding_too_short_encoding_fails(&self) -> Result<(), TestCaseError> {
            let mut encoded = self.encoding.to_owned();
            if encoded.is_empty() {
                return Ok(());
            }
            encoded.pop();
            prop_assert!(T::decode(&encoded).is_err());
            Ok(())
        }

        fn modify_each_element_and_assert_decoding_failure(&self) -> Result<(), TestCaseError> {
            let mut encoding = self.encoding.to_owned();
            for i in 0..encoding.len() {
                let original_value = encoding[i];
                encoding[i] = self.random_encoding[i];
                let decoding = T::decode(&encoding);
                prop_assert!(
                    decoding.is_err() || *decoding.unwrap() != self.value,
                    "failing index: {i}"
                );
                encoding[i] = original_value;
            }
            Ok(())
        }

        fn assert_decoding_random_too_short_encoding_fails_gracefully(
            &self,
        ) -> Result<(), TestCaseError> {
            if self.encoding.is_empty() {
                return Ok(());
            }

            let random_encoding =
                self.random_encoding[..self.length_of_too_short_sequence].to_vec();
            let decoding_result = T::decode(&random_encoding);
            prop_assert!(decoding_result.is_err() || *decoding_result.unwrap() != self.value);
            Ok(())
        }
    }

    #[proptest]
    fn test_encode_decode_random_bfieldelement(
        test_data: BFieldCodecPropertyTestData<BFieldElement>,
    ) {
        test_data.assert_bfield_codec_properties()?;
    }

    #[proptest]
    fn test_encode_decode_random_xfieldelement(
        test_data: BFieldCodecPropertyTestData<XFieldElement>,
    ) {
        test_data.assert_bfield_codec_properties()?;
    }

    #[proptest]
    fn test_encode_decode_random_digest(test_data: BFieldCodecPropertyTestData<Digest>) {
        test_data.assert_bfield_codec_properties()?;
    }

    #[proptest]
    fn test_encode_decode_random_vec_of_bfieldelement(
        test_data: BFieldCodecPropertyTestData<Vec<BFieldElement>>,
    ) {
        test_data.assert_bfield_codec_properties()?;
    }

    #[proptest]
    fn test_encode_decode_random_vec_of_xfieldelement(
        test_data: BFieldCodecPropertyTestData<Vec<XFieldElement>>,
    ) {
        test_data.assert_bfield_codec_properties()?;
    }

    #[proptest]
    fn test_encode_decode_random_vec_of_digest(
        test_data: BFieldCodecPropertyTestData<Vec<Digest>>,
    ) {
        test_data.assert_bfield_codec_properties()?;
    }

    #[proptest]
    fn test_encode_decode_random_vec_of_vec_of_bfieldelement(
        test_data: BFieldCodecPropertyTestData<Vec<Vec<BFieldElement>>>,
    ) {
        test_data.assert_bfield_codec_properties()?;
    }

    #[proptest]
    fn test_encode_decode_random_vec_of_vec_of_xfieldelement(
        test_data: BFieldCodecPropertyTestData<Vec<Vec<XFieldElement>>>,
    ) {
        test_data.assert_bfield_codec_properties()?;
    }

    #[proptest]
    fn test_encode_decode_random_vec_of_vec_of_digest(
        test_data: BFieldCodecPropertyTestData<Vec<Vec<Digest>>>,
    ) {
        test_data.assert_bfield_codec_properties()?;
    }

    #[proptest]
    fn test_encode_decode_tuples_static_static_size_0(
        test_data: BFieldCodecPropertyTestData<(Digest, u128)>,
    ) {
        test_data.assert_bfield_codec_properties()?;
    }

    #[proptest]
    fn test_encode_decode_tuples_static_static_size_1(
        test_data: BFieldCodecPropertyTestData<(Digest, u64)>,
    ) {
        test_data.assert_bfield_codec_properties()?;
    }

    #[proptest]
    fn test_encode_decode_tuples_static_static_size_2(
        test_data: BFieldCodecPropertyTestData<(BFieldElement, BFieldElement)>,
    ) {
        test_data.assert_bfield_codec_properties()?;
    }

    #[proptest]
    fn test_encode_decode_tuples_static_static_size_3(
        test_data: BFieldCodecPropertyTestData<(BFieldElement, XFieldElement)>,
    ) {
        test_data.assert_bfield_codec_properties()?;
    }

    #[proptest]
    fn test_encode_decode_tuples_static_static_size_4(
        test_data: BFieldCodecPropertyTestData<(XFieldElement, BFieldElement)>,
    ) {
        test_data.assert_bfield_codec_properties()?;
    }

    #[proptest]
    fn test_encode_decode_tuples_static_static_size_5(
        test_data: BFieldCodecPropertyTestData<(XFieldElement, Digest)>,
    ) {
        test_data.assert_bfield_codec_properties()?;
    }

    #[proptest]
    fn test_encode_decode_tuples_static_dynamic_size(
        test_data: BFieldCodecPropertyTestData<(Digest, Vec<BFieldElement>)>,
    ) {
        test_data.assert_bfield_codec_properties()?;
    }

    #[proptest]
    fn test_encode_decode_tuples_dynamic_static_size(
        test_data: BFieldCodecPropertyTestData<(Vec<XFieldElement>, Digest)>,
    ) {
        test_data.assert_bfield_codec_properties()?;
    }

    #[proptest]
    fn test_encode_decode_tuples_dynamic_dynamic_size(
        test_data: BFieldCodecPropertyTestData<(Vec<XFieldElement>, Vec<Digest>)>,
    ) {
        test_data.assert_bfield_codec_properties()?;
    }

    #[proptest]
    fn test_phantom_data(test_data: BFieldCodecPropertyTestData<PhantomData<Tip5>>) {
        test_data.assert_bfield_codec_properties()?;
    }

    #[proptest]
    fn test_encode_decode_random_vec_option_xfieldelement(
        test_data: BFieldCodecPropertyTestData<Vec<Option<XFieldElement>>>,
    ) {
        test_data.assert_bfield_codec_properties()?;
    }

    #[proptest]
    fn decode_encode_array_with_static_element_size(
        test_data: BFieldCodecPropertyTestData<[u64; 14]>,
    ) {
        test_data.assert_bfield_codec_properties()?;
    }

    #[proptest]
    fn decode_encode_array_with_dynamic_element_size(
        test_data: BFieldCodecPropertyTestData<[Vec<Digest>; 19]>,
    ) {
        test_data.assert_bfield_codec_properties()?;
    }

    #[test]
    fn static_length_of_array_with_static_element_size_is_as_expected() {
        const N: usize = 14;
        assert_eq!(Some(N * 2), <[u64; N]>::static_length());
    }

    #[test]
    fn static_length_of_array_with_dynamic_element_size_is_as_expected() {
        const N: usize = 19;
        assert!(<[Vec<Digest>; N]>::static_length().is_none());
    }

    #[proptest]
    fn decoding_random_encoding_as_vec_of_bfield_elements_fails(
        random_encoding: Vec<BFieldElement>,
    ) {
        let decoding = Vec::<BFieldElement>::decode(&random_encoding);
        prop_assert!(decoding.is_err());
    }

    #[proptest]
    fn decoding_random_encoding_as_vec_of_xfield_elements_fails(
        random_encoding: Vec<BFieldElement>,
    ) {
        let decoding = Vec::<XFieldElement>::decode(&random_encoding);
        prop_assert!(decoding.is_err());
    }

    #[proptest]
    fn decoding_random_encoding_as_vec_of_digests_fails(random_encoding: Vec<BFieldElement>) {
        let decoding = Vec::<Digest>::decode(&random_encoding);
        prop_assert!(decoding.is_err());
    }

    #[proptest]
    fn decoding_random_encoding_as_vec_of_vec_of_bfield_elements_fails(
        random_encoding: Vec<BFieldElement>,
    ) {
        let decoding = Vec::<Vec<BFieldElement>>::decode(&random_encoding);
        prop_assert!(decoding.is_err());
    }

    #[proptest]
    fn decoding_random_encoding_as_vec_of_vec_of_xfield_elements_fails(
        random_encoding: Vec<BFieldElement>,
    ) {
        let decoding = Vec::<Vec<XFieldElement>>::decode(&random_encoding);
        prop_assert!(decoding.is_err());
    }

    #[cfg(test)]
    pub mod derive_tests {
        // Since we cannot use the derive macro in the same crate where it is defined,
        // we test the macro here instead.

        use arbitrary::Arbitrary;

        use crate::util_types::mmr::mmr_accumulator::MmrAccumulator;
        use crate::{
            shared_math::{digest::Digest, tip5::Tip5, x_field_element::XFieldElement},
            util_types::{
                algebraic_hasher::AlgebraicHasher, mmr::mmr_membership_proof::MmrMembershipProof,
            },
        };

        use super::*;

        #[derive(Debug, Clone, PartialEq, Eq, BFieldCodec, Arbitrary)]
        struct DeriveTestStructA {
            field_a: u64,
            field_b: u64,
            field_c: u64,
        }

        #[test]
        fn bfield_codec_derive_test_struct_a_static_length() {
            assert_eq!(Some(6), DeriveTestStructA::static_length());
        }

        #[proptest]
        fn bfield_codec_derive_test_struct_a(
            test_data: BFieldCodecPropertyTestData<DeriveTestStructA>,
        ) {
            test_data.assert_bfield_codec_properties()?;
        }

        #[derive(Debug, Clone, PartialEq, Eq, BFieldCodec, Arbitrary)]
        struct DeriveTestStructB(u128);

        #[test]
        fn bfield_codec_derive_test_struct_b_static_length() {
            assert_eq!(Some(4), DeriveTestStructB::static_length());
        }

        #[proptest]
        fn bfield_codec_derive_test_struct_b(
            test_data: BFieldCodecPropertyTestData<DeriveTestStructB>,
        ) {
            test_data.assert_bfield_codec_properties()?;
        }

        #[derive(Debug, Clone, PartialEq, Eq, BFieldCodec, Arbitrary)]
        struct DeriveTestStructC(u128, u64, u32);

        #[test]
        fn bfield_codec_derive_test_struct_c_static_length() {
            assert_eq!(Some(7), DeriveTestStructC::static_length());
        }

        #[proptest]
        fn bfield_codec_derive_test_struct_c(
            test_data: BFieldCodecPropertyTestData<DeriveTestStructC>,
        ) {
            test_data.assert_bfield_codec_properties()?;
        }

        // Struct containing Vec<T> where T is BFieldCodec
        #[derive(Debug, Clone, PartialEq, Eq, BFieldCodec, Arbitrary)]
        struct DeriveTestStructD(Vec<u128>);

        #[test]
        fn bfield_codec_derive_test_struct_d_static_length() {
            assert!(DeriveTestStructD::static_length().is_none());
        }

        #[proptest]
        fn bfield_codec_derive_test_struct_d(
            test_data: BFieldCodecPropertyTestData<DeriveTestStructD>,
        ) {
            test_data.assert_bfield_codec_properties()?;
        }

        #[derive(Debug, Clone, PartialEq, Eq, BFieldCodec, Arbitrary)]
        struct DeriveTestStructE(Vec<u128>, u128, u64, Vec<bool>, u32, Vec<u128>);

        #[test]
        fn bfield_codec_derive_test_struct_e_static_length() {
            assert!(DeriveTestStructE::static_length().is_none());
        }

        #[proptest]
        fn bfield_codec_derive_test_struct_e(
            test_data: BFieldCodecPropertyTestData<DeriveTestStructE>,
        ) {
            test_data.assert_bfield_codec_properties()?;
        }

        #[derive(Debug, Clone, PartialEq, Eq, BFieldCodec, Arbitrary)]
        struct DeriveTestStructF {
            field_a: Vec<u64>,
            field_b: bool,
            field_c: u32,
            field_d: Vec<bool>,
            field_e: Vec<BFieldElement>,
            field_f: Vec<BFieldElement>,
        }

        #[test]
        fn bfield_codec_derive_test_struct_f_static_length() {
            assert!(DeriveTestStructF::static_length().is_none());
        }

        #[proptest]
        fn bfield_codec_derive_test_struct_f(
            test_data: BFieldCodecPropertyTestData<DeriveTestStructF>,
        ) {
            test_data.assert_bfield_codec_properties()?;
        }

        #[derive(Debug, Clone, PartialEq, Eq, BFieldCodec, Arbitrary)]
        struct WithPhantomData<H: AlgebraicHasher> {
            a_field: u128,
            #[bfield_codec(ignore)]
            _phantom_data: PhantomData<H>,
            another_field: Vec<u64>,
        }

        #[proptest]
        fn bfield_codec_derive_with_phantom_data(
            test_data: BFieldCodecPropertyTestData<WithPhantomData<Tip5>>,
        ) {
            test_data.assert_bfield_codec_properties()?;
        }

        #[derive(Debug, Clone, PartialEq, Eq, BFieldCodec, Arbitrary)]
        struct WithNestedPhantomData<H: AlgebraicHasher> {
            a_field: u128,
            #[bfield_codec(ignore)]
            _phantom_data: PhantomData<H>,
            another_field: Vec<u64>,
            a_third_field: Vec<WithPhantomData<H>>,
            a_fourth_field: WithPhantomData<Tip5>,
        }

        #[proptest]
        fn bfield_codec_derive_with_nested_phantom_data(
            test_data: BFieldCodecPropertyTestData<WithNestedPhantomData<Tip5>>,
        ) {
            test_data.assert_bfield_codec_properties()?;
        }

        #[derive(Debug, Clone, PartialEq, Eq, BFieldCodec, Arbitrary)]
        struct WithNestedVec {
            a_field: Vec<Vec<u64>>,
        }

        #[test]
        fn bfield_codec_derive_with_nested_vec_static_length() {
            assert!(WithNestedVec::static_length().is_none());
        }

        #[proptest]
        fn bfield_codec_derive_with_nested_vec(
            test_data: BFieldCodecPropertyTestData<WithNestedVec>,
        ) {
            test_data.assert_bfield_codec_properties()?;
        }

        #[derive(Debug, Clone, PartialEq, Eq, BFieldCodec, Arbitrary)]
        struct EmptyStruct {}

        #[proptest]
        fn bfield_codec_derive_empty_struct(test_data: BFieldCodecPropertyTestData<EmptyStruct>) {
            test_data.assert_bfield_codec_properties()?;
        }

        #[derive(Debug, Clone, PartialEq, Eq, BFieldCodec, Arbitrary)]
        struct StructWithEmptyStruct {
            a: EmptyStruct,
        }

        #[proptest]
        fn bfield_codec_derive_struct_with_empty_struct(
            test_data: BFieldCodecPropertyTestData<StructWithEmptyStruct>,
        ) {
            test_data.assert_bfield_codec_properties()?;
        }

        #[derive(Debug, Clone, PartialEq, Eq, BFieldCodec, Arbitrary)]
        struct StructWithTwoEmptyStructs {
            a: EmptyStruct,
            b: EmptyStruct,
        }

        #[proptest]
        fn bfield_codec_derive_struct_with_two_empty_structs(
            test_data: BFieldCodecPropertyTestData<StructWithTwoEmptyStructs>,
        ) {
            test_data.assert_bfield_codec_properties()?;
        }

        #[derive(Debug, Clone, PartialEq, Eq, BFieldCodec, Arbitrary)]
        struct BigStructWithEmptyStructs {
            a: EmptyStruct,
            b: EmptyStruct,
            c: StructWithTwoEmptyStructs,
            d: StructWithEmptyStruct,
            e: EmptyStruct,
        }

        #[proptest]
        fn bfield_codec_derive_big_struct_with_empty_structs(
            test_data: BFieldCodecPropertyTestData<BigStructWithEmptyStructs>,
        ) {
            test_data.assert_bfield_codec_properties()?;
        }

        #[derive(Debug, Clone, PartialEq, Eq, BFieldCodec, Arbitrary)]
        enum EnumWithEmptyStruct {
            A(EmptyStruct),
            B,
            C(EmptyStruct),
        }

        #[proptest]
        fn bfield_codec_derive_enum_with_empty_struct(
            test_data: BFieldCodecPropertyTestData<EnumWithEmptyStruct>,
        ) {
            test_data.assert_bfield_codec_properties()?;
        }

        #[derive(Debug, Clone, PartialEq, Eq, BFieldCodec, Arbitrary)]
        struct MuchNesting {
            a: Vec<Vec<Vec<Vec<BFieldElement>>>>,
            #[bfield_codec(ignore)]
            b: PhantomData<Tip5>,
            #[bfield_codec(ignore)]
            c: PhantomData<Tip5>,
        }

        #[test]
        fn bfield_codec_derive_much_nesting_static_length() {
            assert!(MuchNesting::static_length().is_none());
        }

        #[proptest]
        fn bfield_codec_derive_much_nesting(test_data: BFieldCodecPropertyTestData<MuchNesting>) {
            test_data.assert_bfield_codec_properties()?;
        }

        #[derive(Debug, Clone, PartialEq, Eq, BFieldCodec, Arbitrary)]
        struct SmallArrayStructUnnamedFields([u128; 1]);

        #[test]
        fn bfield_codec_derive_small_array_struct_unnamed_fields_static_length() {
            assert_eq!(Some(4), SmallArrayStructUnnamedFields::static_length());
        }

        #[proptest]
        fn bfield_codec_derive_small_array_struct_unnamed_fields(
            test_data: BFieldCodecPropertyTestData<SmallArrayStructUnnamedFields>,
        ) {
            test_data.assert_bfield_codec_properties()?;
        }

        #[derive(Debug, Clone, PartialEq, Eq, BFieldCodec, Arbitrary)]
        struct SmallArrayStructNamedFields {
            a: [u128; 1],
        }

        #[test]
        fn bfield_codec_derive_small_array_struct_named_fields_static_length() {
            assert_eq!(Some(4), SmallArrayStructNamedFields::static_length());
        }

        #[proptest]
        fn bfield_codec_derive_small_array_struct_named_fields(
            test_data: BFieldCodecPropertyTestData<SmallArrayStructNamedFields>,
        ) {
            test_data.assert_bfield_codec_properties()?;
        }

        #[derive(Debug, Clone, PartialEq, Eq, BFieldCodec, Arbitrary)]
        struct BigArrayStructUnnamedFields([u128; 300]);

        #[test]
        fn bfield_codec_derive_big_array_struct_unnamed_fields_static_length() {
            let array_length = 300;
            let factor = u128::static_length().unwrap();
            let expected_length = Some(array_length * factor);
            assert_eq!(
                expected_length,
                BigArrayStructUnnamedFields::static_length()
            );
        }

        #[proptest(cases = 20)]
        fn bfield_codec_derive_big_array_struct_unnamed_fields(
            test_data: BFieldCodecPropertyTestData<BigArrayStructUnnamedFields>,
        ) {
            test_data.assert_bfield_codec_properties()?;
        }

        #[derive(Debug, Clone, PartialEq, Eq, BFieldCodec, Arbitrary)]
        struct BigArrayStructNamedFields {
            a: [XFieldElement; 300],
            b: XFieldElement,
            c: u64,
        }

        #[test]
        fn bfield_codec_derive_big_array_struct_named_fields_static_length() {
            let array_length = 300;
            let factor = XFieldElement::static_length().unwrap();
            let expected_length = Some(array_length * factor + 3 + 2);
            assert_eq!(expected_length, BigArrayStructNamedFields::static_length());
        }

        #[proptest(cases = 20)]
        fn bfield_codec_derive_big_array_struct_named_fields(
            test_data: BFieldCodecPropertyTestData<BigArrayStructNamedFields>,
        ) {
            test_data.assert_bfield_codec_properties()?;
        }

        #[derive(Debug, Clone, PartialEq, Eq, BFieldCodec, Arbitrary)]
        struct ArrayStructDynamicallySizedElementsUnnamedFields([Vec<u128>; 7]);

        #[test]
        fn bfield_codec_derive_array_struct_dyn_sized_elements_unnamed_fields_static_length() {
            assert!(ArrayStructDynamicallySizedElementsUnnamedFields::static_length().is_none());
        }

        #[proptest]
        fn bfield_codec_derive_array_struct_dyn_sized_elements_unnamed_fields(
            test_data: BFieldCodecPropertyTestData<
                ArrayStructDynamicallySizedElementsUnnamedFields,
            >,
        ) {
            test_data.assert_bfield_codec_properties()?;
        }

        #[derive(Debug, Clone, PartialEq, Eq, BFieldCodec, Arbitrary)]
        struct ArrayStructDynamicallySizedElementsNamedFields {
            a: [Vec<u128>; 7],
        }

        #[test]
        fn bfield_codec_derive_array_struct_dyn_sized_elements_named_fields_static_length() {
            assert!(ArrayStructDynamicallySizedElementsNamedFields::static_length().is_none());
        }

        #[proptest]
        fn bfield_codec_derive_array_struct_dyn_sized_elements_named_fields(
            test_data: BFieldCodecPropertyTestData<ArrayStructDynamicallySizedElementsNamedFields>,
        ) {
            test_data.assert_bfield_codec_properties()?;
        }

        #[derive(Debug, Clone, PartialEq, Eq, BFieldCodec, Arbitrary)]
        struct StructWithTupleField {
            a: (Digest, Vec<Digest>),
        }

        #[test]
        fn bfield_codec_derive_struct_with_tuple_field_static_length() {
            assert!(StructWithTupleField::static_length().is_none());
        }

        #[proptest]
        fn bfield_codec_derive_struct_with_tuple_field(
            test_data: BFieldCodecPropertyTestData<StructWithTupleField>,
        ) {
            test_data.assert_bfield_codec_properties()?;
        }

        #[derive(Debug, Clone, PartialEq, Eq, BFieldCodec, Arbitrary)]
        struct StructWithTupleFieldTwoElements {
            a: ([Digest; 2], XFieldElement),
        }

        #[test]
        fn bfield_codec_derive_struct_with_tuple_field_two_elements_static_length() {
            assert_eq!(Some(13), StructWithTupleFieldTwoElements::static_length());
        }

        #[proptest]
        fn bfield_codec_derive_struct_with_tuple_field_two_elements(
            test_data: BFieldCodecPropertyTestData<StructWithTupleFieldTwoElements>,
        ) {
            test_data.assert_bfield_codec_properties()?;
        }

        #[derive(Debug, Clone, PartialEq, Eq, BFieldCodec, Arbitrary)]
        struct StructWithNestedTupleField {
            a: (([Digest; 2], Vec<XFieldElement>), (XFieldElement, u64)),
        }

        #[test]
        fn bfield_codec_derive_struct_with_nested_tuple_field_static_length() {
            assert!(StructWithNestedTupleField::static_length().is_none());
        }

        #[proptest]
        fn bfield_codec_derive_struct_with_nested_tuple_field(
            test_data: BFieldCodecPropertyTestData<StructWithNestedTupleField>,
        ) {
            test_data.assert_bfield_codec_properties()?;
        }

        #[derive(Debug, Clone, PartialEq, Eq, BFieldCodec, Arbitrary)]
        struct MsMembershipProof<H: AlgebraicHasher> {
            sender_randomness: Digest,
            receiver_preimage: Digest,
            auth_path_aocl: MmrMembershipProof<H>,
        }

        #[test]
        fn bfield_codec_derive_ms_membership_proof_derive_static_length() {
            assert!(MsMembershipProof::<Tip5>::static_length().is_none());
        }

        #[proptest]
        fn bfield_codec_derive_ms_membership_proof_derive(
            test_data: BFieldCodecPropertyTestData<MsMembershipProof<Tip5>>,
        ) {
            test_data.assert_bfield_codec_properties()?;
        }

        #[test]
        fn bfield_codec_derive_mmr_bfieldcodec_derive_static_length() {
            assert!(MmrAccumulator::<Tip5>::static_length().is_none());
        }

        #[proptest]
        fn bfield_codec_derive_mmr_bfieldcodec_derive(
            test_data: BFieldCodecPropertyTestData<MmrAccumulator<Tip5>>,
        ) {
            test_data.assert_bfield_codec_properties()?;
        }

        #[derive(Debug, Clone, PartialEq, Eq, BFieldCodec, Arbitrary)]
        struct UnsupportedFields {
            a: u64,
            #[bfield_codec(ignore)]
            b: usize,
        }

        #[proptest]
        fn unsupported_fields_can_be_ignored_test(#[strategy(arb())] my_struct: UnsupportedFields) {
            let encoded = my_struct.encode();
            let decoded = UnsupportedFields::decode(&encoded).unwrap();
            assert_eq!(
                my_struct.a, decoded.a,
                "Non-ignored fields must be preserved under encoding"
            );
            assert_eq!(
                usize::default(),
                decoded.b,
                "Ignored field must decode to default value"
            )
        }

        #[derive(Debug, Clone, PartialEq, Eq, BFieldCodec, Arbitrary)]
        struct OneFixedLenField {
            some_digest: Digest,
        }

        /// regression test, see https://github.com/Neptune-Crypto/twenty-first/issues/124
        #[proptest]
        fn bfield_codec_derive_vec_of_struct_with_one_fix_len_field_test(
            test_data: BFieldCodecPropertyTestData<Vec<OneFixedLenField>>,
        ) {
            test_data.assert_bfield_codec_properties()?;
        }

        #[derive(Debug, Clone, PartialEq, Eq, BFieldCodec, Arbitrary)]
        pub struct TwoFixedLenFields {
            pub some_digest: Digest,
            pub some_u64: u64,
        }

        #[proptest]
        fn bfield_codec_derive_vec_of_struct_with_two_fix_len_fields_test(
            test_data: BFieldCodecPropertyTestData<Vec<TwoFixedLenFields>>,
        ) {
            test_data.assert_bfield_codec_properties()?;
        }

        #[derive(Debug, Clone, PartialEq, Eq, BFieldCodec, Arbitrary)]
        pub struct TwoFixedLenUnnamedFields(Digest, u64);

        #[proptest]
        fn bfield_codec_derive_vec_of_struct_with_two_fix_len_unnamed_fields_test(
            test_data: BFieldCodecPropertyTestData<Vec<TwoFixedLenUnnamedFields>>,
        ) {
            test_data.assert_bfield_codec_properties()?;
        }

        #[derive(Debug, Clone, PartialEq, Eq, BFieldCodec, Arbitrary)]
        pub struct FixAndVarLenFields {
            pub some_digest: Digest,
            pub some_vec: Vec<u64>,
        }

        #[proptest]
        fn bfield_codec_derive_vec_of_struct_with_fix_and_var_len_fields_test(
            test_data: BFieldCodecPropertyTestData<Vec<FixAndVarLenFields>>,
        ) {
            test_data.assert_bfield_codec_properties()?;
        }

        #[derive(Debug, Clone, PartialEq, Eq, BFieldCodec, Arbitrary)]
        struct FixAndVarLenUnnamedFields(Digest, Vec<u64>);

        #[proptest]
        fn bfield_codec_derive_vec_of_struct_with_fix_and_var_len_unnamed_fields_test(
            test_data: BFieldCodecPropertyTestData<Vec<FixAndVarLenUnnamedFields>>,
        ) {
            test_data.assert_bfield_codec_properties()?;
        }

        #[derive(Debug, Clone, PartialEq, Eq, BFieldCodec, Arbitrary)]
        struct QuiteAFewFixAndVarLenFields {
            some_digest: Digest,
            some_vec: Vec<u64>,
            some_u64: u64,
            other_vec: Vec<u32>,
            other_digest: Digest,
            yet_another_vec: Vec<u32>,
            and_another_vec: Vec<u64>,
            more_fixed_len: u64,
            even_more_fixed_len: u64,
        }

        #[proptest]
        fn bfield_codec_derive_vec_of_struct_with_quite_a_few_fix_and_var_len_fields_test(
            test_data: BFieldCodecPropertyTestData<Vec<QuiteAFewFixAndVarLenFields>>,
        ) {
            test_data.assert_bfield_codec_properties()?;
        }

        #[derive(Debug, Clone, PartialEq, Eq, BFieldCodec, Arbitrary)]
        struct SimpleStructA {
            a: BFieldElement,
            b: XFieldElement,
        }

        #[proptest]
        fn bfield_codec_derive_simple_struct_a(
            test_data: BFieldCodecPropertyTestData<SimpleStructA>,
        ) {
            test_data.assert_bfield_codec_properties()?;
        }

        #[derive(Debug, Clone, PartialEq, Eq, BFieldCodec, Arbitrary)]
        struct SimpleStructB(u32, u64);

        #[proptest]
        fn bfield_codec_derive_simple_struct_b(
            test_data: BFieldCodecPropertyTestData<SimpleStructB>,
        ) {
            test_data.assert_bfield_codec_properties()?;
        }

        #[derive(Debug, Clone, PartialEq, Eq, BFieldCodec, Arbitrary)]
        enum SimpleEnum {
            A,
            B(u32),
            C(BFieldElement, u32, u32),
        }

        #[proptest]
        fn bfield_codec_derive_simple_enum(test_data: BFieldCodecPropertyTestData<SimpleEnum>) {
            test_data.assert_bfield_codec_properties()?;
        }

        #[proptest]
        fn bfield_codec_derive_digest(test_data: BFieldCodecPropertyTestData<Digest>) {
            test_data.assert_bfield_codec_properties()?;
        }

        #[derive(Debug, Clone, PartialEq, Eq, BFieldCodec, Arbitrary)]
        enum ComplexEnum {
            A,
            B(u32),
            C(BFieldElement, XFieldElement, u32),
            D(Vec<BFieldElement>, Digest),
            E(Option<bool>),
        }

        #[proptest]
        fn bfield_codec_derive_complex_enum(test_data: BFieldCodecPropertyTestData<ComplexEnum>) {
            test_data.assert_bfield_codec_properties()?;
        }
    }
}
