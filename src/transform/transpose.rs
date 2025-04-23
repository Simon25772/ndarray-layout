// 引入 crate 中的 ArrayLayout 结构体
use crate::ArrayLayout;
// 引入标准库中的 BTreeSet 用于存储唯一且有序的元素，以及 zip 函数用于迭代多个迭代器
use std::{collections::BTreeSet, iter::zip};

/// 为 ArrayLayout 结构体实现方法
impl<const N: usize> ArrayLayout<N> {
    /// 转置变换允许调换张量的维度顺序，但不改变元素的存储顺序。
    ///
    /// 该方法接收一个排列数组 `perm`，根据该数组重新排列原布局的维度。
    /// 未在 `perm` 中指定的维度将保持不变。
    ///
    /// # 示例
    /// ```rust
    /// # use ndarray_layout::ArrayLayout;
    /// let layout = ArrayLayout::<3>::new(&[2, 3, 4], &[12, 4, 1], 0).transpose(&[1, 0]);
    /// assert_eq!(layout.shape(), &[3, 2, 4]);
    /// assert_eq!(layout.strides(), &[4, 12, 1]);
    /// assert_eq!(layout.offset(), 0);
    /// ```
    ///
    /// # 参数
    /// - `perm`: 一个切片，包含要交换的维度的索引。索引必须唯一。
    ///
    /// # 返回值
    /// 返回一个新的 `ArrayLayout` 实例，其维度顺序已根据 `perm` 进行转置。
    pub fn transpose(&self, perm: &[usize]) -> Self {
        // 将 perm 中的元素收集到 BTreeSet 中，确保元素唯一且有序
        let perm_ = perm.iter().collect::<BTreeSet<_>>();
        // 断言 perm 中的元素都是唯一的
        assert_eq!(perm_.len(), perm.len());

        // 获取当前布局的内容
        let content = self.content();
        // 获取当前布局的形状
        let shape = content.shape();
        // 获取当前布局的步长
        let strides = content.strides();

        // 创建一个新的 ArrayLayout 实例，其维度数量与当前布局相同
        let mut ans = Self::with_ndim(self.ndim);
        // 获取新布局内容的可变引用
        let mut content = ans.content_mut();
        // 将新布局的偏移量设置为当前布局的偏移量
        content.set_offset(self.offset());

        // 定义一个闭包，用于设置新布局指定索引处的形状和步长
        let mut set = |i, j| {
            // 设置新布局索引 i 处的形状为原布局索引 j 处的形状
            content.set_shape(i, shape[j]);
            // 设置新布局索引 i 处的步长为原布局索引 j 处的步长
            content.set_stride(i, strides[j]);
        };

        // 记录上一次处理的维度索引，初始化为 0
        let mut last = 0;
        // 同时遍历有序的 perm_ 和原始的 perm
        for (&i, &j) in zip(perm_, perm) {
            // 处理 last 到 i 之间未在 perm 中指定的维度，保持这些维度不变
            for i in last..i {
                set(i, i);
            }
            // 根据 perm 中的映射关系设置新布局的形状和步长
            set(i, j);
            // 更新 last 为当前处理的维度索引加 1
            last = i + 1;
        }
        // 处理 perm 中未涉及的剩余维度，保持这些维度不变
        for i in last..shape.len() {
            set(i, i);
        }

        // 返回转置后的新布局
        ans
    }
}

/// 测试 transpose 方法的正确性
#[test]
fn test_transpose() {
    // 创建一个初始布局并进行转置操作
    let layout = ArrayLayout::<3>::new(&[2, 3, 4], &[12, 4, 1], 0).transpose(&[1, 0]);
    // 断言转置后的形状是否符合预期
    assert_eq!(layout.shape(), &[3, 2, 4]);
    // 断言转置后的步长是否符合预期
    assert_eq!(layout.strides(), &[4, 12, 1]);
    // 断言转置后的偏移量是否符合预期
    assert_eq!(layout.offset(), 0);

    // 创建另一个初始布局并进行转置操作
    let layout = ArrayLayout::<3>::new(&[2, 3, 4], &[12, 4, 1], 0).transpose(&[2, 0]);
    // 断言转置后的形状是否符合预期
    assert_eq!(layout.shape(), &[4, 3, 2]);
    // 断言转置后的步长是否符合预期
    assert_eq!(layout.strides(), &[1, 4, 12]);
    // 断言转置后的偏移量是否符合预期
    assert_eq!(layout.offset(), 0);
}