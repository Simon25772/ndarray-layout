// 引入 crate 中的 ArrayLayout 结构体，用于后续的广播变换操作
use crate::ArrayLayout;

/// 广播变换参数。该结构体用于存储广播操作所需的信息，包括广播的轴和广播的次数。
#[derive(Clone, PartialEq, Eq, Debug)]
pub struct BroadcastArg {
    /// 广播的轴，指定在哪个维度上进行广播操作。
    pub axis: usize,
    /// 广播次数，即指定轴上的长度要扩增的倍数。
    pub times: usize,
}

/// 为 ArrayLayout 结构体实现广播相关方法
impl<const N: usize> ArrayLayout<N> {
    /// 广播变换将指定的长度为 1 的阶扩增指定的倍数，并将其步长固定为 0。
    /// 广播操作允许在不复制数据的情况下，将一个较小的数组在某个维度上扩展成一个较大的数组。
    ///
    /// # 示例
    /// ```rust
    /// # use ndarray_layout::ArrayLayout;
    /// let layout = ArrayLayout::<3>::new(&[1, 5, 2], &[10, 2, 1], 0).broadcast(0, 10);
    /// assert_eq!(layout.shape(), &[10, 5, 2]);
    /// assert_eq!(layout.strides(), &[0, 2, 1]);
    /// assert_eq!(layout.offset(), 0);
    /// ```
    ///
    /// # 参数
    /// - `axis`: 要进行广播操作的轴的索引。
    /// - `times`: 在指定轴上进行广播的次数，即该轴的新长度。
    ///
    /// # 返回值
    /// 返回一个新的 `ArrayLayout` 实例，其形状和步长已根据广播操作进行更新。
    pub fn broadcast(&self, axis: usize, times: usize) -> Self {
        // 调用 broadcast_many 方法，传入单个广播参数
        self.broadcast_many(&[BroadcastArg { axis, times }])
    }

    /// 一次对多个阶进行广播变换。
    /// 该方法可以同时在多个轴上进行广播操作，提高操作效率。
    ///
    /// # 参数
    /// - `args`: 包含多个 `BroadcastArg` 结构体的切片，每个结构体表示一个轴的广播参数。
    ///
    /// # 返回值
    /// 返回一个新的 `ArrayLayout` 实例，其形状和步长已根据所有广播参数进行更新。
    pub fn broadcast_many(&self, args: &[BroadcastArg]) -> Self {
        // 克隆当前的 ArrayLayout 实例，作为初始的新布局
        let mut ans = self.clone();
        // 获取新布局内容的可变引用，以便修改形状和步长
        let mut content = ans.content_mut();
        // 遍历所有的广播参数
        for &BroadcastArg { axis, times } in args {
            // 断言要广播的轴的原始长度为 1 或者该轴的步长为 0，确保广播操作的合法性
            assert!(content.shape()[axis] == 1 || content.strides()[axis] == 0);
            // 设置指定轴的新形状为广播次数
            content.set_shape(axis, times);
            // 设置指定轴的步长为 0，表示在该轴上广播时不移动数据位置
            content.set_stride(axis, 0);
        }
        // 返回更新后的新布局
        ans
    }
}

/// 测试 broadcast 方法的正确性
#[test]
fn test_broadcast() {
    // 创建一个初始的 ArrayLayout 实例
    let layout = ArrayLayout::<3>::new(&[1, 5, 2], &[10, 2, 1], 0);
    // 对轴 0 进行广播操作，广播次数为 10
    let layout = layout.broadcast(0, 10);
    // 断言广播操作后的形状是否符合预期
    assert_eq!(layout.shape(), &[10, 5, 2]);
    // 断言广播操作后的步长是否符合预期
    assert_eq!(layout.strides(), &[0, 2, 1]);
    // 断言广播操作后的偏移量是否符合预期
    assert_eq!(layout.offset(), 0);
}