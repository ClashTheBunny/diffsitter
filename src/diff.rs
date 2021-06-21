//! Structs and other convenience methods for handling logical concepts pertaining to diffs, such
//! as hunks.

use crate::ast::{EditType, Entry};
use crate::neg_idx_vec::NegIdxVec;
use anyhow::Result;
use std::cmp::Ordering;
use std::collections::VecDeque;
use std::fmt::Debug;
use std::ops::RangeInclusive;
use thiserror::Error;

/// The edit information representing a line
#[derive(Debug, Clone, PartialEq)]
pub struct Line<'a> {
    /// The index of the line in the original document
    pub line_index: usize,
    /// The entries corresponding to the line
    pub entries: VecDeque<Entry<'a>>,
}

impl<'a> Line<'a> {
    pub fn new(line_index: usize) -> Self {
        Line {
            line_index,
            entries: VecDeque::new(),
        }
    }
}

/// A grouping of consecutive edit lines for a document
///
/// Every line in a hunk must be consecutive and in ascending order.
#[derive(Debug, Clone, PartialEq)]
pub struct Hunk<'a>(pub VecDeque<Line<'a>>);

/// Types of errors that come up when inserting an entry to a hunk
#[derive(Debug, Error)]
pub enum HunkInsertionError {
    #[error(
        "Non-adjacent entry (line {incoming_line:?}) added to hunk (last line: {last_line:?})"
    )]
    NonAdjacentHunk {
        incoming_line: usize,
        last_line: usize,
    },
    #[error("Attempted to prepend an entry with a line index ({incoming_line:?}) greater than the first line's index ({first_line:?})")]
    LaterLine {
        incoming_line: usize,
        first_line: usize,
    },
    #[error("Attempted to prepend an entry with a column ({incoming_col:?}) greater than the first entry's column ({first_col:?})")]
    LaterColumn {
        incoming_col: usize,
        first_col: usize,
    },
}

impl<'a> Hunk<'a> {
    /// Create a new, empty hunk
    pub fn new() -> Self {
        Hunk(VecDeque::new())
    }

    /// Returns the first line number of the hunk
    ///
    /// This will return [None] if the internal vector is empty
    pub fn first_line(&self) -> Option<usize> {
        self.0.front().map(|x| x.line_index)
    }

    /// Returns the last line number of the hunk
    ///
    /// This will return [None] if the internal vector is empty
    pub fn last_line(&self) -> Option<usize> {
        self.0.back().map(|x| x.line_index)
    }

    /// Prepend an [entry](Entry) to a hunk
    ///
    /// Entries can only be prepended in descending order (from last to first)
    pub fn push_front(&mut self, entry: Entry<'a>) -> Result<(), HunkInsertionError> {
        let incoming_line_idx = entry.reference.start_position().row;

        // Add a new line vector if the entry has a greater line index, or if the vector is empty.
        // We ensure that the last line has the same line index as the incoming entry.
        if let Some(first_line) = self.0.front() {
            let first_line_idx = first_line.line_index;

            if incoming_line_idx > first_line_idx {
                return Err(HunkInsertionError::LaterLine {
                    incoming_line: incoming_line_idx,
                    first_line: first_line_idx,
                });
            }

            if first_line_idx - incoming_line_idx > 1 {
                return Err(HunkInsertionError::NonAdjacentHunk {
                    incoming_line: incoming_line_idx,
                    last_line: first_line_idx,
                });
            }

            // Only add a new line here if the the incoming line index is one after the last entry
            // If this isn't the case, the incoming line index must be the same as the last line
            // index, so we don't have to add a new line.
            if first_line_idx - incoming_line_idx == 1 {
                self.0.push_front(Line::new(incoming_line_idx));
            }
        } else {
            // line is empty
            self.0.push_front(Line::new(incoming_line_idx));
        }

        // Add the entry to the last line
        let first_line = self.0.front_mut().unwrap();

        // Entries must be added in order, so ensure the last entry in the line has an ending
        // column less than the incoming entry's starting column.
        if let Some(&first_entry) = first_line.entries.back() {
            let first_col = first_entry.reference.end_position().column;
            let incoming_col = entry.reference.end_position().column;

            if incoming_col > first_col {
                return Err(HunkInsertionError::LaterColumn {
                    incoming_col,
                    first_col,
                });
            }
        }
        first_line.entries.push_front(entry);
        Ok(())
    }
}

/// The hunks that correspond to a document
///
/// This type implements a helper builder function that can take
#[derive(Debug, Clone, PartialEq)]
pub struct Hunks<'a>(pub VecDeque<Hunk<'a>>);

impl<'a> Hunks<'a> {
    pub fn new() -> Self {
        Hunks(VecDeque::new())
    }

    /// Push an entry to the front of the hunks
    ///
    /// This will expand the list of hunks if necessary, though the entry must precede the foremost
    /// hunk in the document (by row/column). Failing to do so will result in an error.
    pub fn push_front(&mut self, entry: Entry<'a>) -> Result<()> {
        // If the hunk isn't empty, attempt to prepend an entry into the first hunk
        if let Some(hunk) = self.0.front_mut() {
            let res = hunk.push_front(entry);

            // If the hunk insertion fails because an entry isn't adjacent, then we can create a
            // new hunk. Otherwise we propagate the error since it is a logic error.
            if let Err(HunkInsertionError::NonAdjacentHunk {
                incoming_line: _,
                last_line: _,
            }) = res
            {
                self.0.push_front(Hunk::new());
                self.0.front_mut().unwrap().push_front(entry)?;
            } else {
                res.map_err(|x| anyhow::anyhow!(x))?;
            }
        } else {
            self.0.push_front(Hunk::new());
            self.0.front_mut().unwrap().push_front(entry)?;
        }
        Ok(())
    }
}

/// A difference engine provider
///
/// Any entity that implements this trait is responsible for providing a method
/// that computes the diff between two inputs.
pub trait DiffEngine<'elem, T>
where
    T: Eq + 'elem,
{
    /// The container type to returned from the `diff` function
    type Container;

    /// Compute the shortest edit sequence that will turn `a` into `b`
    fn diff(&self, a: &'elem [T], b: &'elem [T]) -> Self::Container;
}

#[derive(Eq, PartialEq, Copy, Clone, Debug, Default)]
pub struct Myers {}

impl<'elem, T> DiffEngine<'elem, T> for Myers
where
    T: Eq + 'elem + std::fmt::Debug,
{
    type Container = Vec<EditType<&'elem T>>;

    fn diff(&self, a: &'elem [T], b: &'elem [T]) -> Self::Container {
        let mut res = Vec::new();
        // We know the worst case is deleting everything from a and inserting everything from b
        res.reserve(a.len() + b.len());
        Myers::diff_impl(&mut res, a, b);
        res
    }
}

/// Information relevant for a middle snake calculation
#[derive(Debug, PartialEq, Eq, PartialOrd, Ord)]
pub struct MidSnakeInfo {
    /// The index in `a` that corresponds to the middle snake
    pub a_split: i32,

    /// The index in `b` that corresponds to the middle snake
    pub b_split: i32,

    /// the full length of the optimal path between the two inputs
    pub optimal_len: u32,
}

/// Find the difference between two inputs given that you know that they differ exactly by one
/// element. The lengths of the inputs must differ by exactly one.
fn single_difference<'a, T: Eq + 'a>(a: &'a [T], b: &'a [T]) -> &'a T {
    use std::cmp::Ordering::{Equal, Greater, Less};
    assert!(
        ((a.len() as i32) - (b.len() as i32)).abs() == 1,
        "Inputs must differ by exactly 1, got {} and {}",
        a.len(),
        b.len()
    );
    // The insertion/deletion comes from the longer input
    let return_a = a.len() > b.len();

    let mut idx = 0;

    while idx < a.len() && idx < b.len() {
        if a[idx] != b[idx] {
            if return_a {
                return &a[idx];
            } else {
                return &b[idx];
            }
        }
        idx += 1;
    }

    // If there is only one potential difference and everything else matches up, then the only
    // difference can be the last element of the longer input.
    match a.len().cmp(&b.len()) {
        Less => b.last().unwrap(),
        Greater => a.last().unwrap(),
        Equal => unreachable!(),
    }
}

/// Create an inclusive range with unordered elements
///
/// This constructs a proper range regardless of whether the smaller or larger element is passed in
/// any particular order. Normally, if you construct a range, and the lower bound is greater than
/// the upper bound, nothing will ever be included in the range. This method is agnostic to the
/// ordering and will construct the range properly.
fn unordered_range<T: Ord + Copy>(a: T, b: T) -> RangeInclusive<T> {
    let lower_bound = a.min(b);
    let upper_bound = a.max(b);
    RangeInclusive::new(lower_bound, upper_bound)
}

impl Myers {
    fn diff_impl<'elem, T: Eq + Debug + 'elem>(
        res: &mut Vec<EditType<&'elem T>>,
        a: &'elem [T],
        b: &'elem [T],
    ) {
        use Ordering::{Equal, Greater, Less};

        // Some small optimizations: we know if either or both of the inputs are empty, we don't
        // have to bother finding the middle snake
        if a.is_empty() && b.is_empty() {
            return;
        }

        if a.is_empty() {
            for elem in b {
                res.push(EditType::Addition(elem));
            }
            return;
        }

        if b.is_empty() {
            for elem in a {
                res.push(EditType::Deletion(elem));
            }
            return;
        }

        // A single substitution
        /*
        if a.len() == 1 && b.len() == 1 {
            if a[0] != b[0] {
                res.push(EditType::Deletion(&a[0]));
                res.push(EditType::Addition(&b[0]));
            }
            return;
        }
        */

        let MidSnakeInfo {
            optimal_len,
            a_split,
            b_split,
        } = dbg!(Myers::middle_snake(a, b));

        // If the length of the optimal path is more than one, we need to continue dividing and
        // conquering. If the optimal length is <= 1 then b is obtained from a by inserting or
        // deleting at most one symbol, and we know that the shorter of {a, b} is the LCS.
        match optimal_len.cmp(&1) {
            Greater => {
                let first_split = |upper_bound: i32, collection: &'elem [T]| -> &'elem [T] {
                    if upper_bound > -1 {
                        &collection[..=upper_bound as usize]
                    } else {
                        &collection[..0]
                    }
                };
                let second_split = |upper_bound: i32, collection: &'elem [T]| -> &'elem [T] {
                    if upper_bound > -1 {
                        &collection[upper_bound as usize + 1..]
                    } else {
                        &collection[0..]
                    }
                };
                let first_a_split = first_split(a_split, a);
                let first_b_split = first_split(b_split, b);
                let second_a_split = second_split(a_split, a);
                let second_b_split = second_split(b_split, b);

                // TODO(afnan): delete
                println!(
                    "recursive first half:\na: {:#?}\nb: {:#?}",
                    first_a_split, first_b_split
                );
                println!(
                    "recursive second half:\na: {:#?}\nb: {:#?}",
                    second_a_split, second_b_split
                );
                Myers::diff_impl(res, first_a_split, first_b_split);
                Myers::diff_impl(res, second_a_split, second_b_split);
            }
            Equal => {
                debug_assert!(
                    (a.len() as i32 - b.len() as i32).abs() == 1,
                    "a and b should differ by 1"
                );
                // Determine whether the difference is an insertion or a deletion based on which input
                // is longer
                let is_addition = b.len() > a.len();
                let difference = single_difference(a, b);

                if is_addition {
                    res.push(EditType::Addition(difference));
                } else {
                    res.push(EditType::Deletion(difference));
                }
            }
            // if `optimal_len == 0`, we do nothing because there are no edits between a and b
            Less => (),
        }
    }

    /// Find the middle snake that splits an optimal path in half, if one exists
    ///
    /// This implementation directly derives from "An O(ND) Difference Algorithm and Its Variations"
    /// by Myers. This will compute the location of the middle snake and the length of the optimal
    /// shortest edit script.
    // TODO(afnan): need sentinels for (0, 0) and (-1, -1) for proper operation
    pub fn middle_snake<T: Eq>(a: &[T], b: &[T]) -> MidSnakeInfo {
        let n = a.len() as i32;
        let m = b.len() as i32;
        let delta = n - m;
        let is_odd = delta % 2 == 1;
        let midpoint = ((m + n) as f32 / 2.00).ceil() as i32;

        // The size of the frontier vector
        let vec_length = (midpoint * 2) as usize + 1;

        // Checks the bounds of x and y to ensure that they are within the bounds of the inputs in
        // debug builds.
        let check_bounds = |x, y| {
            debug_assert!((0..n).contains(&x), "x={} must be in [0, {})", x, n);
            debug_assert!((0..m).contains(&y), "y={} must be in [0, {})", y, m);
        };

        // We initialize with sentinel index values instead of 0 and n - 1 because there's no guarantee that the first
        // element will match.
        // TODO(afnan): set up proper handling for sentinels
        let mut fwd_front: NegIdxVec<i32> = vec![-1; vec_length].into();
        let mut rev_front: NegIdxVec<i32> = vec![n; vec_length].into();

        for d in 0..=midpoint {
            // Find the end of the furthest reaching forward d-path
            for k in (-d..=d).step_by(2) {
                // k == -d and k != d are just bounds checks to make sure we don't try to compare
                // values outside of the [-d, d] range. We check for the furthest reaching forward
                // frontier by seeing which diagonal has the highest x value.
                let mut x = if k == -d || (k != d && fwd_front[k + 1] >= fwd_front[k - 1]) {
                    // If the longest diagonal is from the vertically connected d - 1 path. The y
                    // component is implicitly added when we compute y below with a different k value.
                    fwd_front[k + 1]
                } else {
                    // If the longest diagonal is from the horizontally connected d - 1 path. We
                    // add one here for the horizontal connection (x, y) -> (x + 1, y).
                    fwd_front[k - 1] + 1
                };
                let mut y = x - k;
                //check_bounds(x, y);

                // Extend the snake
                #[allow(clippy::suspicious_operation_groupings)]
                while x < n && y < m && ((x == -1 && y == -1) || a[x as usize] == b[y as usize]) {
                    x += 1;
                    y += 1;
                }

                fwd_front[k] = x;
                // delta and d can be either negative or positive, so we need to make sure that the
                // range is constructed with `min..=max`, otherwise k will never be in range.
                let delta_range = unordered_range(delta - (d - 1), delta + (d - 1));

                // If delta is odd and k is in the defined range
                if is_odd && delta_range.contains(&k) {
                    // Sanity checks
                    //check_bounds(x, y);

                    // If the path overlaps the furthest reaching reverse d - 1 path in diagonal k
                    // then the length of an SES is 2D - 1, and the last snake of the forward path
                    // is the middle snake.
                    let reverse_x = rev_front[k];

                    // NOTE: that we can convert x and y to `usize` because they are both within
                    // the range of the length of the inputs, which are valid usize values. This property
                    // is also checked with assertions in debug releases.
                    if x >= reverse_x {
                        return MidSnakeInfo {
                            a_split: x,
                            b_split: y,
                            optimal_len: (2 * d - 1) as u32,
                        };
                    }
                }
            }

            // Find the end of the furthest reaching reverse d-path
            for k in (-d..=d).step_by(2) {
                let k = k + delta;

                // k == d and k != -d are just bounds checks to make sure we don't try to compare
                // anything out of range, as explained above. In the reverse path we check to see
                // which diagonal has the *smallest* x value because we're trying to go from the
                // bottom-right to the top-left of the matrix.
                let mut x = if k == d || (k != -d && rev_front[k - 1] <= rev_front[k + 1]) {
                    // If the longest diagonal is from the vertically connected d - 1 path. The y
                    // value is implicitly handled when we compute y with a different k value.
                    rev_front[k - 1]
                } else {
                    // If the longest diagonal is from the horizontally connected d - 1 path. We
                    // subtract x because we're going from (x + 1, y) to (x, y).
                    rev_front[k + 1] - 1
                };
                let mut y = x - k;

                #[allow(clippy::suspicious_operation_groupings)]
                while x > 0
                    && y > 0
                    && ((x == n && y == m) || a[(x - 1) as usize] == b[(y - 1) as usize])
                {
                    x -= 1;
                    y -= 1;
                }
                //check_bounds(x, y);
                rev_front[k] = x;
                let delta_range = unordered_range(-d, d);

                // If delta is even and k is in the defined range, check for an overlap
                if !is_odd && delta_range.contains(&k) {
                    // If the path overlaps with the furthest reaching forward D-path in the
                    // diagonal k + delta, then the length of an SES is 2D and the last snake of
                    // the reverse path is the middle snake.

                    // Sanity checks
                    //check_bounds(x, y);
                    let forward_x = fwd_front[k];

                    // If forward_x >= reverse_x, we have an overlap, so return the furthest
                    // reaching reverse path as the middle snake
                    // NOTE: that we can convert x and y to `usize` because they are both within
                    // the range of the length of the inputs, which are valid usize values.
                    if forward_x >= x {
                        return MidSnakeInfo {
                            a_split: x,
                            b_split: y,
                            optimal_len: (2 * d) as u32,
                        };
                    }
                }
            }
        }
        unreachable!();
    }
}

#[cfg(test)]
mod tests {
    use pretty_assertions::assert_eq;

    use super::*;

    #[test]
    fn test_mid_snake() {
        let input_a = vec![1, 2];
        let input_b = vec![4, 5];
        let mid_snake = Myers::middle_snake(&input_a[..], &input_b[..]);
        let expected = MidSnakeInfo {
            a_split: 0,
            b_split: 1,
            optimal_len: 2,
        };
        assert_eq!(expected, mid_snake);
    }

    fn myers_diff<'a, T: Eq + 'a + Debug>(a: &'a Vec<T>, b: &'a Vec<T>) -> Vec<EditType<&'a T>> {
        let myers = Myers::default();
        myers.diff(&a[..], &b[..])
    }

    #[test]
    fn test_single_diff_empty_deletion() {
        let a: Vec<i32> = vec![0];
        let b: Vec<i32> = Vec::new();
        let diff = single_difference(&a, &b);
        assert_eq!(&a[0], diff);
    }

    #[test]
    fn test_single_diff_empty_addition() {
        let a: Vec<i32> = Vec::new();
        let b: Vec<i32> = vec![0];
        let diff = single_difference(&a, &b);
        assert_eq!(&b[0], diff);
    }

    #[test]
    fn test_single_diff_deletion() {
        let a: Vec<i32> = vec![0, 1, 0];
        let b: Vec<i32> = vec![0, 0];
        let diff = single_difference(&a, &b);
        assert_eq!(&a[1], diff);
    }

    #[test]
    fn test_single_diff_addition() {
        let a: Vec<i32> = vec![0, 0];
        let b: Vec<i32> = vec![0, 1, 0];
        let diff = single_difference(&a, &b);
        assert_eq!(&b[1], diff);
    }

    #[test]
    fn myers_diff_empty_inputs() {
        let input_a: Vec<i32> = vec![];
        let input_b: Vec<i32> = vec![];
        let edit_script = myers_diff(&input_a, &input_b);
        assert!(edit_script.is_empty());
    }

    #[test]
    fn myers_diff_no_diff() {
        let input_a: Vec<i32> = vec![0; 4];
        let input_b: Vec<i32> = vec![0; 4];
        let edit_script = myers_diff(&input_a, &input_b);
        assert!(edit_script.is_empty());
    }

    #[test]
    fn myers_diff_one_addition() {
        let input_a: Vec<i32> = Vec::new();
        let input_b: Vec<i32> = vec![0];
        let expected = vec![EditType::Addition(&input_b[0])];
        let edit_script = myers_diff(&input_a, &input_b);
        assert_eq!(expected, edit_script);
    }

    #[test]
    fn myers_diff_one_deletion() {
        let input_a: Vec<i32> = vec![0];
        let input_b: Vec<i32> = Vec::new();
        let expected = vec![EditType::Deletion(&input_a[0])];
        let edit_script = myers_diff(&input_a, &input_b);
        assert_eq!(expected, edit_script);
    }

    #[test]
    fn myers_diff_single_substitution() {
        let myers = Myers::default();
        let input_a = vec![1];
        let input_b = vec![2];
        let edit_script = myers.diff(&input_a[..], &input_b[..]);
        let expected = vec![
            EditType::Deletion(&input_a[0]),
            EditType::Addition(&input_b[0]),
        ];
        assert_eq!(expected, edit_script);
    }

    #[test]
    fn myers_diff_single_substitution_with_common_elements() {
        let myers = Myers::default();
        let input_a = vec![0, 0, 0];
        let input_b = vec![0, 1, 0];
        let edit_script = myers.diff(&input_a[..], &input_b[..]);
        let expected = vec![EditType::Addition(&input_b[1])];
        assert_eq!(expected, edit_script);
    }
}
