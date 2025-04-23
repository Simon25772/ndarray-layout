// 引入 crate 中的 ArrayLayout 结构体，用于后续的切片操作
use crate::ArrayLayout;
// 引入标准库中的 zip 函数，用于同时迭代多个迭代器
use std::iter::zip;

/// 切片变换参数。该结构体用于存储切片操作所需的信息，包括切片的轴、起始位置、步长和长度。
#[derive(Clone, PartialEq, Eq, Debug)]
pub struct SliceArg {
    /// 切片的轴，指定在哪个维度上进行切片操作。
    pub axis: usize,
    /// 切片的起始位置，即从该轴的哪个位置开始切片。
    pub start: usize,
    /// 切片的步长，决定了切片时元素之间的间隔。正数表示正向切片，负数表示反向切片，0 表示在该位置重复元素。
    pub step: isize,
    /// 切片的长度，即切片操作最终选取的元素数量。
    pub len: usize,
}

/// 为 ArrayLayout 结构体实现切片相关方法
impl<const N: usize> ArrayLayout<N> {
    /// 切片变换是裁剪张量指定阶上一组连续数据的变换。
    ///
    /// 该方法用于在指定的轴上进行切片操作，是 `slice_many` 方法的简化版本，仅对单个轴进行切片。
    ///
    /// # 示例
    /// ```rust
    /// # use ndarray_layout::ArrayLayout;
    /// // 在轴 1 上，从位置 2 开始，步长为 -1，切片长度为 2
    /// let layout = ArrayLayout::<3>::new(&[2, 3, 4], &[12, 4, 1], 0).slice(1, 2, -1, 2);
    /// assert_eq!(layout.shape(), &[2, 2, 4]);
    /// assert_eq!(layout.strides(), &[12, -4, 1]);
    /// assert_eq!(layout.offset(), 8);
    /// ```
    ///
    /// # 参数
    /// - `axis`: 要进行切片的轴的索引。
    /// - `start`: 切片的起始位置。
    /// - `step`: 切片的步长。
    /// - `len`: 切片的长度。
    ///
    /// # 返回值
    /// 返回一个新的 `ArrayLayout` 实例，其形状、步长和偏移量已根据切片操作进行更新。
    pub fn slice(&self, axis: usize, start: usize, step: isize, len: usize) -> Self {
        // 调用 slice_many 方法，传入单个切片参数
        self.slice_many(&[SliceArg {
            axis,
            start,
            step,
            len,
        }])
    }

    /// 一次对多个阶进行切片变换。
    ///
    /// 该方法允许同时在多个轴上进行切片操作，根据传入的 `SliceArg` 切片参数更新布局的形状、步长和偏移量。
    ///
    /// # 参数
    /// - `args`: 包含多个 `SliceArg` 结构体的切片，每个结构体表示一个轴的切片参数。
    ///
    /// # 返回值
    /// 返回一个新的 `ArrayLayout` 实例，其形状、步长和偏移量已根据所有切片参数进行更新。
    pub fn slice_many(&self, mut args: &[SliceArg]) -> Self {
        // 获取当前布局的内容
        let content = self.content();
        // 初始化偏移量为当前布局的偏移量
        let mut offset = content.offset();
        // 同时迭代当前布局的形状和步长，并获取索引
        let iter = zip(content.shape(), content.strides()).enumerate();

        // 创建一个新的 ArrayLayout 实例，其维度数量与当前布局相同
        let mut ans = Self::with_ndim(self.ndim);
        // 获取新布局内容的可变引用
        let mut content = ans.content_mut();
        // 遍历当前布局的形状和步长
        for (i, (&d, &s)) in iter {
            match args {
                // 如果当前轴与切片参数的轴匹配
                [arg, tail @ ..] if arg.axis == i => {
                    // 解构切片参数
                    let &SliceArg {
                        axis,
                        start,
                        step,
                        len,
                    } = arg;
                    // 引入标准库中的 Ordering 枚举，用于比较步长的正负
                    use std::cmp::Ordering::*;
                    // 根据步长的正负计算实际的切片长度
                    let len = match step.cmp(&0) {
                        // 步长为正数的情况
                        Greater => {
                            // 断言起始位置小于该轴的维度大小
                            assert!(start < d);
                            // 更新偏移量
                            offset += start as isize * s;
                            // 计算实际的切片长度
                            (d - start).div_ceil(step as _).min(len)
                        }
                        // 步长为 0 的情况
                        Equal => {
                            // 断言起始位置小于该轴的维度大小
                            assert!(start < d);
                            // 更新偏移量
                            offset += start as isize * s;
                            // 切片长度保持不变
                            len
                        }
                        // 步长为负数的情况
                        Less => {
                            // 确保起始位置不超过该轴的维度大小减 1
                            let start = start.min(d - 1);
                            // 更新偏移量
                            offset += start as isize * s;
                            // 计算实际的切片长度
                            (start + 1).div_ceil((-step) as _).min(len)
                        }
                    };
                    // 设置新布局指定轴的形状为实际的切片长度
                    content.set_shape(i, len);
                    // 设置新布局指定轴的步长为原步长乘以切片步长
                    content.set_stride(i, s * step);

                    // 检查下一个切片参数的轴是否合法
                    if let [next, ..] = tail {
                        assert!(
                            axis < next.axis && next.axis < self.ndim,
                            "next.axis = {} !in ({}, {})",
                            next.axis,
                            axis,
                            self.ndim,
                        );
                    }
                    // 更新剩余的切片参数
                    args = tail;
                }
                // 如果当前轴没有对应的切片参数，保持形状和步长不变
                [..] => {
                    content.set_shape(i, d);
                    content.set_stride(i, s);
                }
            }
        }
        // 设置新布局的偏移量
        content.set_offset(offset as _);
        // 返回新的布局
        ans
    }
}

/// 测试 slice 和 slice_many 方法的正确性
#[test]
fn test_slice() {
    // 测试在轴 1 上，从位置 2 开始，步长为 -1，切片长度为 2 的情况
    let layout = ArrayLayout::<3>::new(&[2, 3, 4], &[12, 4, 1], 0).slice(1, 2, -1, 2);
    assert_eq!(layout.shape(), &[2, 2, 4]);
    assert_eq!(layout.strides(), &[12, -4, 1]);
    assert_eq!(layout.offset(), 8);

    // 测试在轴 1 上，从位置 2 开始，步长为 0，切片长度为 2 的情况
    let layout = ArrayLayout::<3>::new(&[2, 3, 4], &[12, 4, 1], 0).slice(1, 2, 0, 2);
    assert_eq!(layout.shape(), &[2, 2, 4]);
    assert_eq!(layout.strides(), &[12, 0, 1]);
    assert_eq!(layout.offset(), 8);

    // 测试在轴 1 上，从位置 0 开始，步长为 1，切片长度为 2 的情况
    let layout = ArrayLayout::<3>::new(&[2, 3, 4], &[12, 4, 1], 0).slice(1, 0, 1, 2);
    assert_eq!(layout.shape(), &[2, 2, 4]);
    assert_eq!(layout.strides(), &[12, 4, 1]);
    assert_eq!(layout.offset(), 0);

    // 测试同时在轴 1 和轴 2 上进行切片的情况
    let layout = ArrayLayout::<3>::new(&[2, 3, 4], &[12, 4, 1], 0).slice_many(&[SliceArg{
        axis: 1,
        start: 0,
        step: 1,
        len: 2,
    },SliceArg{
        axis: 2,
        start: 0,
        step: 1,    
        len: 4,
    }]);
    assert_eq!(layout.shape(), &[2, 2, 4]);
    assert_eq!(layout.strides(), &[12, 4, 1]);
    assert_eq!(layout.offset(), 0);
}