//! Traits that define the StorageVec interface
//!
//! It is recommended to wildcard import these with
//! `use twenty_first::storage::storage_vec::traits::*`

// use super::iterators::{ManyIterMut, StorageSetter};
use super::{Index, ManyIterMut};
use std::sync::{RwLockReadGuard, RwLockWriteGuard};

// re-export to make life easier for users of our API.
pub use lending_iterator::LendingIterator;

pub trait StorageVecReads<T> {
    /// check if collection is empty
    fn is_empty(&self) -> bool;

    /// get collection length
    fn len(&self) -> Index;

    /// get single element at index
    fn get(&self, index: Index) -> T;

    /// get multiple elements matching indices
    ///
    /// This is a convenience method. For large collections
    /// it may be more efficient to use an iterator or for-loop
    /// and avoid allocating a Vec
    #[inline]
    fn get_many(&self, indices: &[Index]) -> Vec<T> {
        self.many_iter(indices.to_vec()).map(|(_i, v)| v).collect()
    }

    /// get all elements
    ///
    /// This is a convenience method. For large collections
    /// it may be more efficient to use an iterator or for-loop
    /// and avoid allocating a Vec
    #[inline]
    fn get_all(&self) -> Vec<T> {
        self.iter().map(|(_i, v)| v).collect()
    }

    /// get an iterator over all elements
    ///
    /// The returned iterator holds a read-lock over the collection contents.
    /// This enables consistent (snapshot) reads because any writer must
    /// wait until the lock is released.
    ///
    /// The lock is not released until the iterator is dropped, so it is
    /// important to drop the iterator immediately after use.  Typical
    /// for-loop usage does this automatically.
    ///
    /// If a write is attempted before the read lock is dropped, a deadlock
    /// will occur.
    ///
    /// Correct Example:
    /// ```
    /// for (key, value) in vec.iter() {
    ///     println!("{key}: {value}")
    /// } // <--- iterator is dropped here.
    ///
    /// // write can proceed
    /// let val = vec.put(5, 2);
    /// ```
    ///
    /// Deadlock Example:
    /// ```
    /// let iter = vec.iter();
    /// while let Some(key, val) = iter.next() {
    ///     println!("{key}: {value}")
    /// }
    ///
    /// // deadlock! This will wait for the write lock forever because iter is still holding read lock.
    /// let val = vec.put(5, 2);
    /// ```
    ///
    /// note: any write op would deadlock, including `iter_mut()`, `many_iter_mut()`, `set_many()`, etc.
    #[inline]
    fn iter(&self) -> Box<dyn Iterator<Item = (Index, T)> + '_> {
        self.many_iter(0..self.len())
    }

    /// The returned iterator holds a read-lock over the collection contents.
    /// This enables consistent (snapshot) reads because any writer must
    /// wait until the lock is released.
    ///
    /// The lock is not released until the iterator is dropped, so it is
    /// important to drop the iterator immediately after use.  Typical
    /// for-loop usage does this automatically.
    ///
    /// If a write is attempted before the read lock is dropped, a deadlock
    /// will occur.
    ///
    /// Correct Example:
    /// ```
    /// for (key, value) in vec.iter_values() {
    ///     println!("{value}")
    /// } // <--- iterator is dropped here.
    ///
    /// // write can proceed
    /// let val = vec.push(2);
    /// ```
    ///
    /// Deadlock Example:
    /// ```
    /// let iter = vec.iter();
    /// while let Some(val) = iter.next() {
    ///     println!("{value}")
    /// }
    ///
    /// // deadlock! This will wait for the write lock forever because iter is still holding read lock.
    /// let val = vec.push(2);
    /// ```
    ///
    /// note: any write op would deadlock, including `iter_mut()`, `many_iter_mut()`, `set_many()`, etc.
    #[inline]
    fn iter_values(&self) -> Box<dyn Iterator<Item = T> + '_> {
        self.many_iter_values(0..self.len())
    }

    /// get an iterator over elements matching indices
    ///
    /// The returned iterator holds a read-lock over the collection contents.
    /// This enables consistent (snapshot) reads because any writer must
    /// wait until the lock is released.
    ///
    /// The lock is not released until the iterator is dropped, so it is
    /// important to drop the iterator immediately after use.  Typical
    /// for-loop usage does this automatically.
    ///
    /// If a write is attempted before the read lock is dropped, a deadlock
    /// will occur.
    ///
    /// Correct Example:
    /// ```
    /// for (key, value) in vec.many_iter([3, 5, 7]) {
    ///     println!("{key}: {value}")
    /// } // <--- iterator is dropped here.
    ///
    /// // write can proceed
    /// let val = vec.put(5, 2);
    /// ```
    ///
    /// Deadlock Example:
    /// ```
    /// let iter = vec.many_iter([3, 5, 7]);
    /// while let Some(key, val) = iter.next() {
    ///     println!("{key}: {value}")
    /// }
    ///
    /// // deadlock! This will wait for the write lock forever because iter is still holding read lock.
    /// let val = vec.put(5, 2);
    /// ```
    ///
    /// note: any write op would deadlock, including `iter_mut()`, `many_iter_mut()`, `set_many()`, etc.
    fn many_iter(
        &self,
        indices: impl IntoIterator<Item = Index> + 'static,
    ) -> Box<dyn Iterator<Item = (Index, T)> + '_>;

    /// get an iterator over elements matching indices
    ///
    /// The returned iterator holds a read-lock over the collection contents.
    /// This enables consistent (snapshot) reads because any writer must
    /// wait until the lock is released.
    ///
    /// The lock is not released until the iterator is dropped, so it is
    /// important to drop the iterator immediately after use.  Typical
    /// for-loop usage does this automatically.
    ///
    /// If a write is attempted before the read lock is dropped, a deadlock
    /// will occur.
    ///
    /// Correct Example:
    /// ```
    /// for (key, value) in vec.many_iter_values([2, 5, 8]) {
    ///     println!("{value}")
    /// } // <--- iterator is dropped here.
    ///
    /// // write can proceed
    /// let val = vec.put(5, 2);
    /// ```
    ///
    /// Deadlock Example:
    /// ```
    /// let iter = vec.many_iter_values([2, 5, 8]);
    /// while let Some(val) = iter.next() {
    ///     println!("{value}")
    /// }
    ///
    /// // deadlock! This will wait for the write lock forever because iter is still holding read lock.
    /// let val = vec.put(5, 2);
    /// ```
    ///
    /// note: any write op would deadlock, including `iter_mut()`, `many_iter_mut()`, `set_many()`, etc.
    fn many_iter_values(
        &self,
        indices: impl IntoIterator<Item = Index> + 'static,
    ) -> Box<dyn Iterator<Item = T> + '_>;
}

pub trait StorageVecImmutableWrites<T>: StorageVecReads<T> {
    // type LockedData;

    /// set a single element.
    ///
    /// note: The update is performed as a single atomic operation.
    fn set(&self, index: Index, value: T);

    /// set multiple elements.
    ///
    /// note: all updates are performed as a single atomic operation.
    ///       readers will see either the before or after state,
    ///       never an intermediate state.
    fn set_many(&self, key_vals: impl IntoIterator<Item = (Index, T)>) {
        for (key, val) in key_vals.into_iter() {
            self.set(key, val)
        }
    }

    /// set elements from start to vals.count()
    ///
    /// note: all updates are performed as a single atomic operation.
    ///       readers will see either the before or after state,
    ///       never an intermediate state.
    #[inline]
    fn set_first_n(&self, vals: impl IntoIterator<Item = T>) {
        self.set_many((0..).zip(vals));
    }

    /// set all elements with a simple list of values in an array or Vec
    /// and validates that input length matches target length.
    ///
    /// panics if input length does not match target length.
    ///
    /// note: all updates are performed as a single atomic operation.
    ///       readers will see either the before or after state,
    ///       never an intermediate state.
    ///
    /// note: casts the input value's length from usize to Index
    ///       so will panic if vals contains more than 2^32 items
    #[inline]
    fn set_all(&self, vals: impl IntoIterator<IntoIter = impl ExactSizeIterator<Item = T>>) {
        let iter = vals.into_iter();

        assert!(
            iter.len() as Index == self.len(),
            "size-mismatch.  input has {} elements and target has {} elements.",
            iter.len(),
            self.len(),
        );

        self.set_first_n(iter);
    }

    /// pop an element from end of collection
    ///
    /// note: The update is performed as a single atomic operation.
    fn pop(&self) -> Option<T>;

    /// push an element to end of collection
    ///
    /// note: The update is performed as a single atomic operation.
    fn push(&self, value: T);

    /// get a mutable iterator over all elements
    ///
    /// note: all updates are performed as a single atomic operation.
    ///       readers will see either the before or after state,
    ///       never an intermediate state.
    ///
    /// note: the returned (lending) iterator cannot be used in a for loop.  Use a
    ///       while loop instead.  See example below.
    ///
    /// Important: The returned iterator holds a write lock over `StorageVecRwLock::LockedData`.
    /// This write lock must be dropped before performing any read operation or the
    /// code will deadlock.  See Deadlock Example.
    ///
    /// Correct Example:
    /// ```
    /// {
    ///     let mut iter = vec.iter_mut();
    ///         while let Some(mut setter) = iter.next() {
    ///         setter.set(50);
    ///     }
    /// } // <----- iter is dropped here.  write lock is released.
    ///
    /// // read can proceed
    /// let val = vec.get(2);
    /// ```
    ///
    /// Deadlock Example:
    /// ```
    /// let mut iter = vec.iter_mut();
    /// while let Some(mut setter) = iter.next() {
    ///     setter.set(50);
    /// }
    ///
    /// // deadlock! This will wait for the read lock forever because iter is still holding write lock.
    /// let val = vec.get(2);
    /// ```
    ///
    /// note: any read op would deadlock, including `iter()`, `many_iter()`, `get_many()`, etc.
    #[allow(private_bounds)]
    #[inline]
    fn iter_mut(&self) -> ManyIterMut<Self, T>
    where
        Self: Sized + StorageVecRwLock<T>,
    {
        ManyIterMut::new(0..self.len(), self)
    }

    /// get a mutable iterator over elements matching indices
    ///
    /// note: all updates are performed as a single atomic operation.
    ///       readers will see either the before or after state,
    ///       never an intermediate state.
    ///
    /// note: the returned (lending) iterator cannot be used in a for loop.  Use a
    ///       while loop instead.  See example below.
    ///
    /// Important: The returned iterator holds a write lock over `StorageVecRwLock::LockedData`.
    /// This write lock must be dropped before performing any read operation or the
    /// code will deadlock.  See Deadlock Example.
    ///
    /// Correct Example:
    /// ```
    /// {
    ///     let mut iter = vec.many_iter_mut([2, 4, 6]);
    ///         while let Some(mut setter) = iter.next() {
    ///         setter.set(50);
    ///     }
    /// } // <----- iter is dropped here.  write lock is released.
    ///
    /// // read can proceed
    /// let val = vec.get(2);
    /// ```
    ///
    /// Deadlock Example:
    /// ```
    /// let mut iter = vec.many_iter_mut([2, 4, 6]);
    /// while let Some(mut setter) = iter.next() {
    ///     setter.set(50);
    /// }
    ///
    /// // deadlock! This will wait for the read lock forever because iter is still holding write lock.
    /// let val = vec.get(2);
    /// ```
    ///
    /// note: any read op would deadlock, including `iter()`, `many_iter()`, `get_many()`, etc.
    #[allow(private_bounds)]
    #[inline]
    fn many_iter_mut(
        &self,
        indices: impl IntoIterator<Item = Index> + 'static,
    ) -> ManyIterMut<Self, T>
    where
        Self: Sized + StorageVecRwLock<T>,
    {
        ManyIterMut::new(indices, self)
    }
}

// We keep this trait private so that the locks remain encapsulated inside our API.
pub(in super::super) trait StorageVecRwLock<T> {
    type LockedData;

    /// obtain write lock over mutable data.
    fn write_lock(&self) -> RwLockWriteGuard<Self::LockedData>;

    /// obtain read lock over mutable data.
    fn read_lock(&self) -> RwLockReadGuard<Self::LockedData>;
}

pub(in super::super) trait StorageVecIterMut<T>: StorageVec<T> {}

pub trait StorageVecMutableWrites<T>: StorageVecReads<T> {
    /// set a single element.
    fn set(&mut self, index: Index, value: T);

    /// set multiple elements.
    fn set_many(&mut self, key_vals: impl IntoIterator<Item = (Index, T)>) {
        for (key, val) in key_vals.into_iter() {
            self.set(key, val)
        }
    }

    /// set elements from start to vals.count()
    #[inline]
    fn set_first_n(&mut self, vals: impl IntoIterator<Item = T>) {
        self.set_many((0..).zip(vals));
    }

    /// set all elements with a simple list of values in an array or Vec
    /// and validates that input length matches target length.
    ///
    /// calls ::set_many() internally.
    ///
    /// panics if input length does not match target length.
    ///
    /// note: casts the input value's length from usize to Index
    ///       so will panic if vals contains more than 2^32 items
    #[inline]
    fn set_all(&mut self, vals: impl IntoIterator<IntoIter = impl ExactSizeIterator<Item = T>>) {
        let iter = vals.into_iter();

        assert!(
            iter.len() as Index == self.len(),
            "size-mismatch.  input has {} elements and target has {} elements.",
            iter.len(),
            self.len(),
        );

        self.set_first_n(iter);
    }

    /// pop an element from end of collection
    fn pop(&mut self) -> Option<T>;

    /// push an element to end of collection
    fn push(&mut self, value: T);
}

pub trait StorageVec<T>: StorageVecReads<T> + StorageVecImmutableWrites<T> {}