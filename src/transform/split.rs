// 引入 crate 中的 ArrayLayout 结构体
use crate::ArrayLayout;

/// 切分变换参数。该结构体用于存储切分操作所需的信息，以便将一个 `ArrayLayout` 沿指定维度切分成多个部分。
///
/// - `src`: 指向要进行切分操作的原始 `ArrayLayout` 的引用。
/// - `axis`: 指定要进行切分的维度的索引。
/// - `start`: 当前切分部分在指定维度上的起始位置。
/// - `parts`: 一个切片，包含每个切分部分在指定维度上的大小。
pub struct Split<'a, const N: usize> {
    // 要进行切分的原始 ArrayLayout 的引用
    src: &'a ArrayLayout<N>,
    // 进行切分的维度
    axis: usize,
    // 当前切分的起始位置
    start: usize,
    // 每个切分部分的大小
    parts: &'a [usize],
}

/// 为 ArrayLayout 结构体实现切分相关方法
impl<const N: usize> ArrayLayout<N> {
    /// 切分变换将单个张量沿某个维度切分成多个张量，因此可以支持不均匀的切分。
    ///
    /// 该方法返回一个 `Split` 迭代器，用于逐个获取切分后的 `ArrayLayout` 实例。
    ///
    /// # 示例
    /// ```rust
    /// # use ndarray_layout::ArrayLayout;
    /// let layout = ArrayLayout::<3>::new(&[2, 3, 4], &[12, 4, 1], 0);
    /// let mut splits = layout.split(2, &[1, 3]);
    ///
    /// let layout = splits.next().unwrap();
    /// assert_eq!(layout.shape(), &[2, 3, 1]);
    /// assert_eq!(layout.strides(), &[12, 4, 1]);
    /// assert_eq!(layout.offset(), 0);
    ///
    /// let layout = splits.next().unwrap();
    /// assert_eq!(layout.shape(), &[2, 3, 3]);
    /// assert_eq!(layout.strides(), &[12, 4, 1]);
    /// assert_eq!(layout.offset(), 1);
    /// ```
    ///
    /// # 参数
    /// - `axis`: 要进行切分的维度的索引。
    /// - `parts`: 一个切片，包含每个切分部分在指定维度上的大小。所有部分大小之和必须等于指定维度的原始大小。
    ///
    /// # 返回值
    /// 返回一个 `Split` 迭代器，用于遍历切分后的 `ArrayLayout` 实例。
    #[inline]
    pub fn split<'a>(&'a self, axis: usize, parts: &'a [usize]) -> Split<'a, N> {
        // 断言指定维度的原始大小等于所有切分部分大小之和
        assert_eq!(self.shape()[axis], parts.iter().sum());
        // 创建并返回 Split 结构体实例
        Split {
            src: self,
            axis,
            start: 0,
            parts,
        }
    }
}

/// 为 Split 结构体实现 Iterator trait，使其可以作为迭代器使用
impl<const N: usize> Iterator for Split<'_, N> {
    // 迭代器返回的元素类型为 ArrayLayout<N>
    type Item = ArrayLayout<N>;

    /// 获取迭代器的下一个元素。
    ///
    /// 该方法会从 `parts` 中取出第一个元素作为当前切分部分的大小，
    /// 然后根据当前的起始位置和切分大小生成一个新的 `ArrayLayout` 实例。
    ///
    /// # 返回值
    /// - 如果 `parts` 不为空，返回 `Some(ArrayLayout<N>)`，表示下一个切分后的 `ArrayLayout` 实例。
    /// - 如果 `parts` 为空，返回 `None`，表示迭代结束。
    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        // 尝试从 parts 中取出第一个元素和剩余部分
        self.parts.split_first().map(|(&head, tail)| {
            // 记录当前的起始位置
            let start = self.start;
            // 更新起始位置为当前起始位置加上当前切分部分的大小
            self.start += head;
            // 更新 parts 为剩余部分
            self.parts = tail;
            // 调用 src 的 slice 方法生成切分后的 ArrayLayout 实例
            self.src.slice(self.axis, start, 1, head)
        })
    }
}

/// 测试 split 方法的正确性
#[test]
fn test_split() {
    // 创建一个 ArrayLayout 实例
    let layout = ArrayLayout::<3>::new(&[2, 3, 4], &[12, 4, 1], 0);
    // 调用 split 方法进行切分，得到一个 Split 迭代器
    let mut splits = layout.split(2, &[1, 3]);
    // 获取第一个切分后的 ArrayLayout 实例
    let layout = splits.next().unwrap();
    // 断言第一个切分后的形状是否符合预期
    assert_eq!(layout.shape(), &[2, 3, 1]);
    // 断言第一个切分后的步长是否符合预期
    assert_eq!(layout.strides(), &[12, 4, 1]);
    // 断言第一个切分后的偏移量是否符合预期
    assert_eq!(layout.offset(), 0);
    // 获取第二个切分后的 ArrayLayout 实例
    let layout = splits.next().unwrap();
    // 断言第二个切分后的形状是否符合预期
    assert_eq!(layout.shape(), &[2, 3, 3]);
    // 断言第二个切分后的步长是否符合预期
    assert_eq!(layout.strides(), &[12, 4, 1]);
    // 断言第二个切分后的偏移量是否符合预期
    assert_eq!(layout.offset(), 1);
}