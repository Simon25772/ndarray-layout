// 引入 crate 中的 ArrayLayout 结构体，用于后续的索引变换操作
use crate::ArrayLayout;
// 引入标准库中的 zip 函数，用于同时迭代多个迭代器
use std::iter::zip;

/// 索引变换参数。该结构体用于存储索引变换所需的信息，包括索引的轴和选择的元素索引。
#[derive(Clone, PartialEq, Eq, Debug)]
pub struct IndexArg {
    /// 索引的轴，指定在哪个维度上进行索引操作。
    pub axis: usize,
    /// 选择指定轴的第几个元素，索引从 0 开始。
    pub index: usize,
}

/// 为 ArrayLayout 结构体实现索引相关方法
impl<const N: usize> ArrayLayout<N> {
    /// 索引变换是选择张量指定阶上一项数据的变换，例如指定向量中的一个数、指定矩阵的一行或一列。
    /// 索引变换导致张量降阶，确定索引的阶从张量表示移除。
    ///
    /// # 示例
    /// ```rust
    /// # use ndarray_layout::ArrayLayout;
    /// let layout = ArrayLayout::<3>::new(&[2, 3, 4], &[12, 4, 1], 0).index(1, 2);
    /// assert_eq!(layout.shape(), &[2, 4]);
    /// assert_eq!(layout.strides(), &[12, 1]);
    /// assert_eq!(layout.offset(), 8);
    /// ```
    ///
    /// # 参数
    /// - `axis`: 要进行索引操作的轴的索引。
    /// - `index`: 在指定轴上选择的元素的索引。
    ///
    /// # 返回值
    /// 返回一个新的 `ArrayLayout` 实例，其形状、步长和偏移量已根据索引操作进行更新，维度数减少。
    pub fn index(&self, axis: usize, index: usize) -> Self {
        // 调用 index_many 方法，传入单个索引参数
        self.index_many(&[IndexArg { axis, index }])
    }

    /// 一次对多个阶进行索引变换。
    ///
    /// # 参数
    /// - `args`: 包含多个 `IndexArg` 结构体的切片，每个结构体表示一个轴的索引参数。
    ///
    /// # 返回值
    /// 返回一个新的 `ArrayLayout` 实例，其形状、步长和偏移量已根据所有索引参数进行更新，维度数减少。
    pub fn index_many(&self, mut args: &[IndexArg]) -> Self {
        // 获取当前布局的内容
        let content = self.content();
        // 初始化偏移量为当前布局的偏移量
        let mut offset = content.offset();
        // 获取当前布局的形状
        let shape = content.shape();
        // 同时迭代当前布局的形状和步长，并获取索引
        let iter = zip(shape, content.strides()).enumerate();

        // 定义一个闭包，用于检查索引参数是否有效
        let check = |&IndexArg { axis, index }| shape.get(axis).filter(|&&d| index < d).is_some();

        // 检查第一个索引参数是否有效，如果无效则触发断言失败
        if let [first, ..] = args {
            assert!(check(first), "Invalid index arg: {first:?}");
        } else {
            // 如果没有索引参数，直接克隆当前布局并返回
            return self.clone();
        }

        // 创建一个新的 ArrayLayout 实例，其维度数量为当前布局的维度数量减去索引参数的数量
        let mut ans = Self::with_ndim(self.ndim - args.len());
        // 获取新布局内容的可变引用
        let mut content = ans.content_mut();
        // 初始化新布局的索引
        let mut j = 0;
        // 遍历当前布局的形状和步长
        for (i, (&d, &s)) in iter {
            match *args {
                // 如果当前轴与索引参数的轴匹配
                [IndexArg { axis, index }, ref tail @ ..] if axis == i => {
                    // 根据索引更新偏移量
                    offset += index as isize * s;
                    // 检查下一个索引参数是否有效
                    if let [first, ..] = tail {
                        assert!(check(first), "Invalid index arg: {first:?}");
                        // 确保索引参数的轴按升序排列
                        assert!(first.axis > axis, "Index args must be in ascending order");
                    }
                    // 更新剩余的索引参数
                    args = tail;
                }
                // 如果当前轴没有对应的索引参数，将形状和步长设置到新布局中
                [..] => {
                    content.set_shape(j, d);
                    content.set_stride(j, s);
                    j += 1;
                }
            }
        }
        // 设置新布局的偏移量
        content.set_offset(offset as _);
        // 返回新的布局
        ans
    }
}

/// 测试 index 和 index_many 方法的正确性
#[test]
fn test() {
    // 错误：这里应该是 ArrayLayout::<3>，修正后创建一个三维数组布局
    let layout = ArrayLayout::<3>::new(&[2, 3, 4], &[12, 4, 1], 0);
    // 对轴 1 进行索引操作，选择第 2 个元素
    let layout = layout.index(1, 2);
    // 断言索引操作后的形状是否符合预期
    assert_eq!(layout.shape(), &[2, 4]);
    // 断言索引操作后的步长是否符合预期
    assert_eq!(layout.strides(), &[12, 1]);
    // 断言索引操作后的偏移量是否符合预期
    assert_eq!(layout.offset(), 8);

    // 错误：这里应该是 ArrayLayout::<3>，修正后创建一个三维数组布局
    let layout = ArrayLayout::<3>::new(&[2, 3, 4], &[12, -4, 1], 20);
    // 对轴 1 进行索引操作，选择第 2 个元素
    let layout = layout.index(1, 2);
    // 断言索引操作后的形状是否符合预期
    assert_eq!(layout.shape(), &[2, 4]);
    // 断言索引操作后的步长是否符合预期
    assert_eq!(layout.strides(), &[12, 1]);
    // 断言索引操作后的偏移量是否符合预期
    assert_eq!(layout.offset(), 12);

    // 错误：这里应该是 ArrayLayout::<3>，修正后创建一个三维数组布局
    let layout = ArrayLayout::<3>::new(&[2, 3, 4], &[12, -4, 1], 20);
    // 调用 index_many 方法，传入空的索引参数切片
    let layout = layout.index_many(&[]);
    // 断言索引操作后的形状是否符合预期
    assert_eq!(layout.shape(), &[2, 3, 4]);
    // 断言索引操作后的步长是否符合预期
    assert_eq!(layout.strides(), &[12, -4, 1]);
    // 断言索引操作后的偏移量是否符合预期
    assert_eq!(layout.offset(), 20);

    // 错误：这里应该是 ArrayLayout::<3>，修正后创建一个三维数组布局
    let layout = ArrayLayout::<3>::new(&[2, 3, 4], &[12, -4, 1], 20);
    // 调用 index_many 方法，传入多个索引参数
    let layout = layout.index_many(&[
        IndexArg {
            axis: 0,
            index: 1,
        },
        IndexArg {
            axis: 1,
            index: 2,
        },
    ]);
    // 断言索引操作后的形状是否符合预期
    assert_eq!(layout.shape(), &[4]);
    // 断言索引操作后的步长是否符合预期
    assert_eq!(layout.strides(), &[1]);
    // 断言索引操作后的偏移量是否符合预期
    assert_eq!(layout.offset(), 24);
}