use crate::{ArrayLayout, Endian};
use std::iter::zip;

/// 合并变换参数。
#[derive(Clone, PartialEq, Eq, Debug)]
pub struct MergeArg {
    /// 合并的起点。
    pub start: usize,
    /// 合并的宽度
    pub len: usize,
    /// 分块的顺序。
    pub endian: Option<Endian>,
}

impl<const N: usize> ArrayLayout<N> {
    /// 合并变换是将多个连续维度划分合并的变换。
    /// 大端合并对维度从后到前依次合并。
    ///
    /// ```rust
    /// # use ndarray_layout::ArrayLayout;
    /// let layout = ArrayLayout::<3>::new(&[2, 3, 4], &[12, 4, 1], 0).merge_be(0, 3).unwrap();
    /// assert_eq!(layout.shape(), &[24]);
    /// assert_eq!(layout.strides(), &[1]);
    /// assert_eq!(layout.offset(), 0);
    /// ```
    #[inline]
    pub fn merge_be(&self, start: usize, len: usize) -> Option<Self> {
        self.merge_many(&[MergeArg {
            start,
            len,
            endian: Some(Endian::BigEndian),
        }])
    }

    /// 合并变换是将多个连续维度划分合并的变换。
    /// 小端合并对维度从前到后依次合并。
    ///
    /// ```rust
    /// # use ndarray_layout::ArrayLayout;
    /// let layout = ArrayLayout::<3>::new(&[4, 3, 2], &[1, 4, 12], 0).merge_le(0, 3).unwrap();
    /// assert_eq!(layout.shape(), &[24]);
    /// assert_eq!(layout.strides(), &[1]);
    /// assert_eq!(layout.offset(), 0);
    /// ```
    #[inline]
    pub fn merge_le(&self, start: usize, len: usize) -> Option<Self> {
        self.merge_many(&[MergeArg {
            start,
            len,
            endian: Some(Endian::LittleEndian),
        }])
    }

    /// 合并变换是将多个连续维度划分合并的变换。
    /// 任意合并只考虑维度的存储连续性。
    ///
    /// ```rust
    /// # use ndarray_layout::ArrayLayout;
    /// let layout = ArrayLayout::<3>::new(&[3, 2, 4], &[4, 12, 1], 0).merge_free(0, 3).unwrap();
    /// assert_eq!(layout.shape(), &[24]);
    /// assert_eq!(layout.strides(), &[1]);
    /// assert_eq!(layout.offset(), 0);
    /// ```
    #[inline]
    pub fn merge_free(&self, start: usize, len: usize) -> Option<Self> {
        self.merge_many(&[MergeArg {
            start,
            len,
            endian: None,
        }])
    }

    /// 一次对多个阶进行合并变换。
    pub fn merge_many(&self, args: &[MergeArg]) -> Option<Self> {
        let content = self.content();
        let shape = content.shape();
        let strides = content.strides();

        let merged = args.iter().map(|arg| arg.len).sum::<usize>();
        let mut ans = Self::with_ndim(self.ndim + args.len() - merged);

        let mut content = ans.content_mut();
        content.set_offset(self.offset());
        let mut i = 0;
        let mut push = |d, s| {
            content.set_shape(i, d);
            content.set_stride(i, s);
            i += 1;
        };

        let mut last_end = 0;
        for arg in args {
            let &MergeArg { start, len, endian } = arg;
            let end = start + len;

            if len == 0 {
                continue;
            }

            for j in last_end..arg.start {
                push(shape[j], strides[j]);
            }

            let mut pairs = zip(&shape[start..end], &strides[start..end]).collect::<Vec<_>>();
            match endian {
                Some(Endian::BigEndian) => pairs.reverse(),
                Some(Endian::LittleEndian) => {}
                None => pairs.sort_unstable_by_key(|(_, &s)| s.unsigned_abs()),
            }

            let ((&d, &s), pairs) = pairs.split_first().unwrap();
            let mut d = d;

            for (&d_, &s_) in pairs {
                // 合并的维度长度若有 0 或 1 则不需要判断步长
                if d <= 1 || d_ <= 1 || s_ == s * d as isize {
                    d *= d_
                } else {
                    return None;
                }
            }

            push(d, s);
            last_end = end;
        }
        for j in last_end..shape.len() {
            push(shape[j], strides[j]);
        }

        Some(ans)
    }
}
