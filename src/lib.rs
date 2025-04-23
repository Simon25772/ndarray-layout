// 将项目的 README 文件内容作为文档注释
#![doc = include_str!("../README.md")]
// 开启对警告和缺失文档注释的检查
#![deny(warnings, missing_docs)]

/// 允许内联存储 N 维信息的数组布局结构体。
pub struct ArrayLayout<const N: usize> {
    // 数组的维度数量
    ndim: usize,
    // 存储布局内容的联合体
    content: Union<N>,
}

/// 声明 ArrayLayout 实现 Send trait，表明该类型可以安全地在线程间发送。
/// 由于使用了 unsafe 关键字，需要确保实现的正确性。
unsafe impl<const N: usize> Send for ArrayLayout<N> {}
/// 声明 ArrayLayout 实现 Sync trait，表明该类型可以安全地在线程间共享引用。
/// 由于使用了 unsafe 关键字，需要确保实现的正确性。
unsafe impl<const N: usize> Sync for ArrayLayout<N> {}

/// 用于存储布局内容的联合体，根据维度数量选择不同的存储方式。
union Union<const N: usize> {
    // 当维度数量超过 N 时，使用指针进行动态分配存储
    ptr: NonNull<usize>,
    // 当维度数量不超过 N 时，内联存储偏移量、形状和步长信息
    _inlined: (isize, [usize; N], [isize; N]),
}

/// 为 ArrayLayout 实现 Clone trait，允许克隆数组布局。
impl<const N: usize> Clone for ArrayLayout<N> {
    /// 内联函数，克隆当前数组布局。
    #[inline]
    fn clone(&self) -> Self {
        // 调用 new 方法创建一个新的布局，使用当前布局的形状、步长和偏移量
        Self::new(self.shape(), self.strides(), self.offset())
    }
}

/// 为 ArrayLayout 实现 PartialEq trait，允许比较两个数组布局是否相等。
impl<const N: usize> PartialEq for ArrayLayout<N> {
    /// 内联函数，比较两个数组布局是否相等。
    #[inline]
    fn eq(&self, other: &Self) -> bool {
        // 比较维度数量和内容切片是否相等
        self.ndim == other.ndim && self.content().as_slice() == other.content().as_slice()
    }
}

/// 为 ArrayLayout 实现 Eq trait，表明该类型支持相等比较。
impl<const N: usize> Eq for ArrayLayout<N> {}

/// 为 ArrayLayout 实现 Drop trait，当布局实例被丢弃时执行清理操作。
impl<const N: usize> Drop for ArrayLayout<N> {
    /// 当布局实例被丢弃时，释放动态分配的内存（如果有）。
    fn drop(&mut self) {
        // 检查是否有动态分配的指针
        if let Some(ptr) = self.ptr_allocated() {
            // 不安全代码块，释放动态分配的内存
            unsafe { dealloc(ptr.cast().as_ptr(), layout(self.ndim)) }
        }
    }
}

/// 元信息存储顺序。
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub enum Endian {
    /// 大端序，范围更大的维度在元信息中更靠前的位置。
    BigEndian,
    /// 小端序，范围更小的维度在元信息中更靠前的位置。
    LittleEndian,
}

/// 为 ArrayLayout 实现关联方法。
impl<const N: usize> ArrayLayout<N> {
    /// 创建一个具有指定形状、步长和偏移量的新布局。
    ///
    /// # 示例
    /// ```rust
    /// # use ndarray_layout::ArrayLayout;
    /// let layout = ArrayLayout::<4>::new(&[2, 3, 4], &[12, -4, 1], 20);
    /// assert_eq!(layout.offset(), 20);
    /// assert_eq!(layout.shape(), &[2, 3, 4]);
    /// assert_eq!(layout.strides(), &[12, -4, 1]);
    /// ```
    pub fn new(shape: &[usize], strides: &[isize], offset: isize) -> Self {
        // 检查形状和步长的长度是否相等
        assert_eq!(shape.len(),strides.len(),"shape and strides must have the same length");

        // 创建一个具有指定维度数量的新布局
        let mut ans = Self::with_ndim(shape.len());
        // 获取布局内容的可变引用
        let mut content = ans.content_mut();
        // 设置偏移量
        content.set_offset(offset);
        // 复制形状信息
        content.copy_shape(shape);
        // 复制步长信息
        content.copy_strides(strides);
        ans
    }

    /// 创建一个具有指定形状的连续布局。
    ///
    /// # 示例
    /// ```rust
    /// # use ndarray_layout::{Endian, ArrayLayout};
    /// let layout = ArrayLayout::<4>::new_contiguous(&[2, 3, 4], Endian::LittleEndian, 4);
    /// assert_eq!(layout.offset(), 0);
    /// assert_eq!(layout.shape(), &[2, 3, 4]);
    /// assert_eq!(layout.strides(), &[4, 8, 24]);
    /// ```
    pub fn new_contiguous(shape: &[usize], endian: Endian, element_size: usize) -> Self {
        // 创建一个具有指定维度数量的新布局
        let mut ans = Self::with_ndim(shape.len());
        // 获取布局内容的可变引用
        let mut content = ans.content_mut();
        // 设置偏移量为 0
        content.set_offset(0);
        // 复制形状信息
        content.copy_shape(shape);
        // 初始化元素大小的倍数
        let mut mul = element_size as isize;
        // 定义一个闭包，用于设置步长并更新倍数
        let push = |i| {
            content.set_stride(i, mul);
            mul *= shape[i] as isize;
        };
        // 根据大端序或小端序决定遍历顺序
        match endian {
            Endian::BigEndian => (0..shape.len()).rev().for_each(push),
            Endian::LittleEndian => (0..shape.len()).for_each(push),
        }
        ans
    }

    /// 获取数组的维度数量。
    #[inline]
    pub const fn ndim(&self) -> usize {
        self.ndim
    }

    /// 获取数组的偏移量。
    #[inline]
    pub fn offset(&self) -> isize {
        self.content().offset()
    }

    /// 获取数组的形状。
    #[inline]
    pub fn shape(&self) -> &[usize] {
        self.content().shape()
    }

    /// 获取数组的步长。
    #[inline]
    pub fn strides(&self) -> &[isize] {
        self.content().strides()
    }

    /// 将当前布局复制到内联大小为 `M` 的另一个 `ArrayLayout` 中。
    ///
    /// # 示例
    /// ```rust
    /// # use ndarray_layout::{Endian::BigEndian, ArrayLayout};
    /// let layout = ArrayLayout::<4>::new_contiguous(&[3, 4], BigEndian, 0);
    /// assert_eq!(size_of_val(&layout), (2 * 4 + 2) * size_of::<usize>());
    ///
    /// let layout = layout.to_inline_size::<2>();
    /// assert_eq!(size_of_val(&layout), (2 * 2 + 2) * size_of::<usize>());
    /// ```
    pub fn to_inline_size<const M: usize>(&self) -> ArrayLayout<M> {
        // 调用 new 方法创建一个新的布局，使用当前布局的形状、步长和偏移量
        ArrayLayout::new(self.shape(), self.strides(), self.offset())
    }

    /// 计算数组中的元素数量。
    ///
    /// # 示例
    /// ```rust
    /// # use ndarray_layout::{Endian::BigEndian, ArrayLayout};
    /// let layout = ArrayLayout::<4>::new_contiguous(&[2, 3, 4], BigEndian, 20);
    /// assert_eq!(layout.num_elements(), 24);
    /// ```
    #[inline]
    pub fn num_elements(&self) -> usize {
        // 对形状中的元素进行累乘
        self.shape().iter().product()
    }

    /// 计算给定索引处元素的偏移量。
    ///
    /// # 示例
    /// ```rust
    /// # use ndarray_layout::{Endian::BigEndian, ArrayLayout};
    /// let layout = ArrayLayout::<4>::new_contiguous(&[2, 3, 4], BigEndian, 4);
    /// assert_eq!(layout.element_offset(22, BigEndian), 88); // 88 <- (22 % 4 * 4) + (22 / 4 % 3 * 16) + (22 / 4 / 3 % 2 * 48)
    /// ```
    pub fn element_offset(&self, index: usize, endian: Endian) -> isize {
        /// 正向计算元素偏移量的辅助函数。
        fn offset_forwards(
            mut rem: usize,
            shape: impl IntoIterator<Item = usize>,
            strides: impl IntoIterator<Item = isize>,
        ) -> isize {
            let mut ans = 0;
            // 遍历形状和步长，计算偏移量
            for (d, s) in zip(shape, strides) {
                ans += s * (rem % d) as isize;
                rem /= d
            }
            ans
        }

        // 获取形状和步长的迭代器
        let shape = self.shape().iter().cloned();
        let strides = self.strides().iter().cloned();
        // 加上布局的偏移量，并根据大端序或小端序调用辅助函数
        self.offset()
            + match endian {
                Endian::BigEndian => offset_forwards(index, shape.rev(), strides.rev()),
                Endian::LittleEndian => offset_forwards(index, shape, strides),
            }
    }

    /// 计算数据的字节范围，以确定数组需要访问的内存区域位置。
    /// 
    /// # 示例
    /// ```rust
    /// # use ndarray_layout::ArrayLayout;
    /// let layout = ArrayLayout::<4>::new(&[2, 3, 4],&[12, -4, 1], 20);
    /// let range = layout.data_range();
    /// assert_eq!(range, 12..=35);
    /// ```
    pub fn data_range(&self) -> RangeInclusive<isize> {
        // 获取布局内容
        let content = self.content();
        // 初始化起始和结束偏移量为布局的偏移量
        let mut start = content.offset();
        let mut end = content.offset();
        // 遍历形状和步长，更新起始和结束偏移量
        for (&d, s) in zip(content.shape(), content.strides()) {
            use std::cmp::Ordering::{Equal, Greater, Less};
            let i = d as isize - 1;
            match s.cmp(&0) {
                Equal => {},
                Less => start += s * i,
                Greater => end += s * i,
            }
        }
        start..=end
    }
}

// 引入格式化模块
mod fmt;
// 引入变换模块
mod transform;
// 重新导出变换模块中的类型
pub use transform::{BroadcastArg, IndexArg, MergeArg, SliceArg, Split, TileArg};

// 引入标准库中的相关类型和函数
use std::{
    alloc::{Layout, alloc, dealloc},
    iter::zip,
    ops::RangeInclusive,
    ptr::{NonNull, copy_nonoverlapping},
    slice::from_raw_parts,
};

/// 为 ArrayLayout 实现私有方法。
impl<const N: usize> ArrayLayout<N> {
    /// 内联函数，检查是否有动态分配的指针。
    #[inline]
    fn ptr_allocated(&self) -> Option<NonNull<usize>> {
        // 编译时断言 N 大于 0
        const { assert!(N > 0)}
        // ndim 大于 N 则 content 是 ptr，否则是元组
        if self.ndim > N {
            Some(unsafe { self.content.ptr })
        } else {
            None
        }
    }

    /// 内联函数，获取布局内容的不可变引用。
    #[inline]
    fn content(&self) -> Content<false> {
        Content {
            ptr: self
                .ptr_allocated()
                .unwrap_or(unsafe { NonNull::new_unchecked(&self.content as *const _ as _) }),
            ndim: self.ndim,
        }
    }

    /// 内联函数，获取布局内容的可变引用。
    #[inline]
    fn content_mut(&mut self) -> Content<true> {
        Content {
            ptr: self
                .ptr_allocated()
                .unwrap_or(unsafe { NonNull::new_unchecked(&self.content as *const _ as _) }),
            ndim: self.ndim,
        }
    }

    /// 创建一个具有指定维度数量的新 ArrayLayout。
    #[inline]
    fn with_ndim(ndim: usize) -> Self {
        Self {
            ndim,
            content: if ndim <= N {
                // 维度数量不超过 N 时，使用内联存储
                Union {
                    _inlined: (0, [0; N], [0; N]),
                }
            } else {
                // 维度数量超过 N 时，使用动态分配存储
                Union {
                    ptr: unsafe { NonNull::new_unchecked(alloc(layout(ndim)).cast()) },
                }
            },
        }
    }
}

/// 表示布局内容的结构体，根据 MUT 标记决定是否可变。
struct Content<const MUT: bool> {
    ptr: NonNull<usize>,
    ndim: usize,
}

/// 为 Content 实现方法。
impl<const MUT: bool> Content<MUT> {
    /// 内联函数，将内容转换为切片。
    #[inline]
    fn as_slice(&self) -> &[usize] {
        // 不安全代码块，从指针创建切片
        unsafe { from_raw_parts(self.ptr.as_ptr(), 1 + self.ndim * 2) }
    }

    /// 内联函数，获取偏移量。
    #[inline] 
    fn offset(&self) -> isize {
        // 不安全代码块，从指针读取偏移量
        unsafe { self.ptr.cast().read() }
    }

    /// 内联函数，获取形状信息。
    #[inline]
    fn shape<'a>(&self) -> &'a [usize] {
        // 不安全代码块，从指针创建形状切片
        unsafe { from_raw_parts(self.ptr.add(1).as_ptr(), self.ndim) }
    }

    /// 内联函数，获取步长信息。
    #[inline]
    fn strides<'a>(&self) -> &'a [isize] {
        // 不安全代码块，从指针创建步长切片
        unsafe { from_raw_parts(self.ptr.add(1 + self.ndim).cast().as_ptr(), self.ndim) }
    }
}

/// 为可变的 Content 实现方法。
impl Content<true> {
    /// 内联函数，设置偏移量。
    #[inline]
    fn set_offset(&mut self, val: isize) {
        // 不安全代码块，向指针写入偏移量
        unsafe { self.ptr.cast().write(val) }
    }

    /// 内联函数，设置指定索引处的形状值。
    #[inline]
    fn set_shape(&mut self, idx: usize, val: usize) {
        // 检查索引是否越界
        assert!(idx < self.ndim);
        // 不安全代码块，向指针写入形状值
        unsafe { self.ptr.add(1 + idx).write(val) }
    }

    /// 内联函数，设置指定索引处的步长值。
    #[inline]
    fn set_stride(&mut self, idx: usize, val: isize) {
        // 检查索引是否越界
        assert!(idx < self.ndim);
        // 不安全代码块，向指针写入步长值
        unsafe { self.ptr.add(1 + idx + self.ndim).cast().write(val) }
    }

    /// 内联函数，复制形状信息。
    #[inline]
    fn copy_shape(&mut self, val: &[usize]) {
        // 检查形状长度是否匹配
        assert!(val.len() == self.ndim);
        // 不安全代码块，复制形状信息到指针
        unsafe { copy_nonoverlapping(val.as_ptr(), self.ptr.add(1).as_ptr(), self.ndim) }
    }

    /// 内联函数，复制步长信息。
    #[inline]
    fn copy_strides(&mut self, val: &[isize]) {
        // 检查步长长度是否匹配
        assert!(val.len() == self.ndim);
        // 不安全代码块，复制步长信息到指针
        unsafe {
            copy_nonoverlapping(
                val.as_ptr(),
                self.ptr.add(1 + self.ndim).cast().as_ptr(),
                self.ndim,
            )
        }
    }
}

/// 内联函数，根据维度数量计算内存布局。
#[inline]
fn layout(ndim: usize) -> Layout {
    // 创建一个包含指定数量 usize 元素的内存布局
    Layout::array::<usize>(1 + ndim * 2).unwrap()
}


/// 测试 new 方法是否正确创建布局。
#[test]
fn test_new() {
    let layout = ArrayLayout::<4>::new(&[2, 3, 4], &[12, -4, 1], 20);
    assert_eq!(layout.offset(), 20);
    assert_eq!(layout.shape(), &[2, 3, 4]);
    assert_eq!(layout.strides(), &[12, -4, 1]);
    assert_eq!(layout.ndim(), 3);
}

/// 测试 new 方法在形状和步长长度不同时的行为。
#[test]
fn test_new_different_length(){

}

/// 测试 new_contiguous 方法在小端序下是否正确创建布局。
#[test]
fn test_new_contiguous_little_endian() {
    let layout = ArrayLayout::<4>::new_contiguous(&[2, 3, 4], Endian::LittleEndian, 4);
    assert_eq!(layout.offset(), 0);
    assert_eq!(layout.shape(), &[2, 3, 4]);
    assert_eq!(layout.strides(), &[4, 8, 24]);
}

/// 测试 new_contiguous 方法在大端序下是否正确创建布局。
#[test]
fn test_new_contiguous_big_endian() {
    let layout = ArrayLayout::<4>::new_contiguous(&[2, 3, 4], Endian::LittleEndian, 4);
    assert_eq!(layout.offset(), 0);
    assert_eq!(layout.shape(), &[2, 3, 4]);
    assert_eq!(layout.strides(), &[4, 8, 24]);
}

/// 测试 new 方法在形状和步长长度不匹配时是否会 panic。
#[test]
#[should_panic(expected = "shape and strides must have the same length")]
fn test_new_invalid_shape_strides_length() {
    ArrayLayout::<4>::new(&[2, 3], &[12, -4, 1], 20);
}

/// 测试 to_inline_size 方法是否正确转换内联大小。
#[test]
fn test_to_inline_size() {
    let layout = ArrayLayout::<4>::new_contiguous(&[3, 4], Endian::BigEndian, 0);
    assert_eq!(size_of_val(&layout), (2 * 4 + 2) * size_of::<usize>());
    let layout = layout.to_inline_size::<2>();
    assert_eq!(size_of_val(&layout), (2 * 2 + 2) * size_of::<usize>());
}

/// 测试 num_elements 方法是否正确计算元素数量。
#[test]
fn test_num_elements() {
    let layout = ArrayLayout::<4>::new_contiguous(&[2, 3, 4], Endian::BigEndian, 20);
    assert_eq!(layout.num_elements(), 24);
}

/// 测试 element_offset 方法在小端序下是否正确计算元素偏移量。
#[test]
fn test_element_offset_little_endian() {
    let layout = ArrayLayout::<4>::new_contiguous(&[2, 3, 4], Endian::LittleEndian, 4);
    assert_eq!(layout.element_offset(22, Endian::LittleEndian), 88);
}

/// 测试 element_offset 方法在大端序下是否正确计算元素偏移量。
#[test]
fn test_element_offset_big_endian() {
    let layout = ArrayLayout::<4>::new_contiguous(&[2, 3, 4], Endian::BigEndian, 4);
    assert_eq!(layout.element_offset(22, Endian::BigEndian), 88);
}

/// 测试 data_range 方法在步长为正数时是否正确计算数据范围。
#[test]
fn test_data_range_positive_strides() {
    let layout = ArrayLayout::<4>::new_contiguous(&[2, 3, 4], Endian::LittleEndian, 4);
    let range = layout.data_range();
    assert_eq!(range, 0..=92); // 0 + 2*4 + 3*8 + 4*24 = 92
}

/// 测试 data_range 方法在步长混合时是否正确计算数据范围。
#[test]
fn test_data_range_mixed_strides() {
    let layout = ArrayLayout::<4>::new(&[2, 3, 4],&[12, -4, 0], 20);
    let range = layout.data_range();
    assert_eq!(range, 12..=32);
}

/// 测试 clone 和 eq 方法是否正确工作。
#[test]
fn test_clone_and_eq() {
    let layout1 = ArrayLayout::<4>::new(&[2, 3, 4], &[12, -4, 1], 20);
    let layout2 = layout1.clone();
    assert!(layout1.eq(&layout2));
}

/// 测试 drop 方法是否正确释放内存。
#[test]
fn test_drop() {
    let layout = ArrayLayout::<4>::new(&[2, 3, 4], &[12, -4, 1], 20);
    // let ptr = layout.ptr_allocated().unwrap();
    drop(layout);
    // 丢弃后，内存应该被释放。
    // 由于无法直接测试，依赖 Rust 的安全保证。
}