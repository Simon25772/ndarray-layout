// 引入 crate 中的 ArrayLayout 结构体和 Endian 枚举
use crate::{ArrayLayout, Endian};
// 引入标准库中的 zip 函数，用于同时迭代多个迭代器
use std::iter::zip;

/// 合并变换参数。该结构体用于存储合并操作所需的信息，包括合并的起始位置、合并的维度数量以及分块顺序。
#[derive(Clone, PartialEq, Eq, Debug)]
pub struct MergeArg {
    /// 合并的起点，即从哪个维度开始进行合并操作。
    pub start: usize,
    /// 合并的宽度，即要合并的连续维度的数量。
    pub len: usize,
    /// 分块的顺序。`Some(Endian::BigEndian)` 表示大端合并，`Some(Endian::LittleEndian)` 表示小端合并，`None` 表示任意合并。
    pub endian: Option<Endian>,
}

/// 为 ArrayLayout 结构体实现合并相关方法
impl<const N: usize> ArrayLayout<N> {
    /// 合并变换是将多个连续维度划分合并的变换。
    /// 大端合并对维度从后到前依次合并。
    ///
    /// # 示例
    /// ```rust
    /// # use ndarray_layout::ArrayLayout;
    /// let layout = ArrayLayout::<3>::new(&[2, 3, 4], &[12, 4, 1], 0).merge_be(0, 3).unwrap();
    /// assert_eq!(layout.shape(), &[24]);
    /// assert_eq!(layout.strides(), &[1]);
    /// assert_eq!(layout.offset(), 0);
    /// ```
    ///
    /// # 参数
    /// - `start`: 合并操作的起始维度索引。
    /// - `len`: 要合并的连续维度的数量。
    ///
    /// # 返回值
    /// 如果合并成功，返回 `Some(ArrayLayout<N>)`；否则返回 `None`。
    #[inline]
    pub fn merge_be(&self, start: usize, len: usize) -> Option<Self> {
        // 调用 merge_many 方法，传入大端合并的参数
        self.merge_many(&[MergeArg {
            start,
            len,
            endian: Some(Endian::BigEndian),
        }])
    }

    /// 合并变换是将多个连续维度划分合并的变换。
    /// 小端合并对维度从前到后依次合并。
    ///
    /// # 示例
    /// ```rust
    /// # use ndarray_layout::ArrayLayout;
    /// let layout = ArrayLayout::<3>::new(&[4, 3, 2], &[1, 4, 12], 0).merge_le(0, 3).unwrap();
    /// assert_eq!(layout.shape(), &[24]);
    /// assert_eq!(layout.strides(), &[1]);
    /// assert_eq!(layout.offset(), 0);
    /// ```
    ///
    /// # 参数
    /// - `start`: 合并操作的起始维度索引。
    /// - `len`: 要合并的连续维度的数量。
    ///
    /// # 返回值
    /// 如果合并成功，返回 `Some(ArrayLayout<N>)`；否则返回 `None`。
    #[inline]
    pub fn merge_le(&self, start: usize, len: usize) -> Option<Self> {
        // 调用 merge_many 方法，传入小端合并的参数
        self.merge_many(&[MergeArg {
            start,
            len,
            endian: Some(Endian::LittleEndian),
        }])
    }

    /// 合并变换是将多个连续维度划分合并的变换。
    /// 任意合并只考虑维度的存储连续性。
    ///
    /// # 示例
    /// ```rust
    /// # use ndarray_layout::ArrayLayout;
    /// let layout = ArrayLayout::<3>::new(&[3, 2, 4], &[4, 12, 1], 0).merge_free(0, 3).unwrap();
    /// assert_eq!(layout.shape(), &[24]);
    /// assert_eq!(layout.strides(), &[1]);
    /// assert_eq!(layout.offset(), 0);
    /// ```
    ///
    /// # 参数
    /// - `start`: 合并操作的起始维度索引。
    /// - `len`: 要合并的连续维度的数量。
    ///
    /// # 返回值
    /// 如果合并成功，返回 `Some(ArrayLayout<N>)`；否则返回 `None`。
    #[inline]
    pub fn merge_free(&self, start: usize, len: usize) -> Option<Self> {
        // 调用 merge_many 方法，传入任意合并的参数
        self.merge_many(&[MergeArg {
            start,
            len,
            endian: None,
        }])
    }

    /// 一次对多个阶进行合并变换。
    ///
    /// # 参数
    /// - `args`: 包含多个 `MergeArg` 结构体的切片，每个结构体表示一组合并操作的参数。
    ///
    /// # 返回值
    /// 如果所有合并操作都成功，返回 `Some(ArrayLayout<N>)`；否则返回 `None`。
    pub fn merge_many(&self, args: &[MergeArg]) -> Option<Self> {
        // 获取当前布局的内容
        let content = self.content();
        // 获取当前布局的形状
        let shape = content.shape();
        // 获取当前布局的步长
        let strides = content.strides();

        // 修改 BUG：计算合并后的维度数量，确保每个合并操作至少合并 1 个维度
        let merged = args.iter().map(|arg| arg.len.max(1)).sum::<usize>();
        // 创建一个新的 ArrayLayout 实例，计算新的维度数量
        let mut ans = Self::with_ndim(self.ndim + args.len() - merged);

        // 获取新布局内容的可变引用
        let mut content = ans.content_mut();
        // 将新布局的偏移量设置为当前布局的偏移量
        content.set_offset(self.offset());
        // 初始化新布局的索引
        let mut i = 0;
        // 定义一个闭包，用于设置新布局的形状和步长
        let mut push = |d, s| {
            content.set_shape(i, d);
            content.set_stride(i, s);
            i += 1;
        };

        // 记录上一次合并操作的结束位置
        let mut last_end = 0;
        // 遍历所有合并操作的参数
        for arg in args {
            // 解构合并操作的参数
            let &MergeArg { start, len, endian } = arg;
            // 计算本次合并操作的结束位置
            let end = start + len;

            // 如果合并的宽度为 0，跳过本次合并操作
            if len == 0 {
                continue;
            }

            // 将上一次合并操作结束位置到本次合并操作起始位置之间的维度添加到新布局中
            for j in last_end..arg.start {
                push(shape[j], strides[j]);
            }

            // 创建一个向量，用于存储要合并的维度的形状和步长对
            let mut pairs = Vec::with_capacity(len);
            // 遍历要合并的维度，将非 0 和非 1 的维度添加到向量中
            for (&d, &s) in zip(&shape[start..end], &strides[start..end]) {
                match d {
                    0 => todo!(), // 处理维度大小为 0 的情况，目前待实现
                    1 => {} // 忽略维度大小为 1 的情况
                    _ => pairs.push((d, s)), // 将非 0 和非 1 的维度添加到向量中
                }
            }

            // 修改 BUG：更新上一次合并操作的结束位置
            last_end = end;

            // 如果向量为空，说明要合并的维度都是 0 或 1，添加一个形状为 1，步长为 0 的维度
            if pairs.is_empty() {
                push(1, 0);
                continue;
            }

            // 根据合并的顺序对向量进行排序或反转
            match endian {
                Some(Endian::BigEndian) => pairs.reverse(), // 大端合并，反转向量
                Some(Endian::LittleEndian) => {} // 小端合并，不做处理
                None => pairs.sort_unstable_by_key(|(_, s)| s.unsigned_abs()), // 任意合并，按步长的绝对值排序
            }

            // 取出向量的第一个元素
            let ((d, s), pairs) = pairs.split_first().unwrap();
            // 初始化合并后的维度大小
            let mut d = *d;

            // 遍历剩余的元素，检查步长是否符合合并条件
            for &(d_, s_) in pairs {
                if s_ == s * d as isize {
                    d *= d_ // 如果符合条件，更新合并后的维度大小
                } else {
                    return None; // 不符合条件，合并失败，返回 None
                }
            }

            // 将合并后的维度添加到新布局中
            push(d, *s);
        }

        // 将最后一次合并操作结束位置到原布局末尾的维度添加到新布局中
        for j in last_end..shape.len() {
            push(shape[j], strides[j]);
        }

        // 返回合并后的新布局
        Some(ans)
    }
}

/// 测试 merge_be 方法在合并失败时返回 None
#[test]
fn test_merge_return_none() {
    // 创建一个三维数组布局
    let layout = ArrayLayout::<3>::new(&[16, 4, 2], &[8, 4, 1], 0);
    // 尝试从第 0 个维度开始合并 3 个维度
    let merged_layout = layout.merge_be(0, 3);
    // 断言合并操作失败，返回 None
    assert!(merged_layout.is_none());
}

/// 测试当要合并的维度对为空时的合并操作
#[test]
fn test_merge_pairs_empyt(){
    // 创建一个三维数组布局
    let layout = ArrayLayout::<3>::new(&[1, 1, 1], &[1, 1, 1], 0);
    // 尝试从第 0 个维度开始合并 2 个维度
    let merged_layout = layout.merge_be(0, 2).unwrap();
    // 断言合并后的形状符合预期
    assert_eq!(merged_layout.shape(), &[1, 1]);
    // 断言合并后的步长符合预期
    assert_eq!(merged_layout.strides(), &[0, 1]);
    // 断言合并后的偏移量符合预期
    assert_eq!(merged_layout.offset(), 0);
}

/// 测试 merge_be 方法的示例用法
#[test]
fn test_merge_be_example() {
    // 创建一个三维数组布局
    let layout = ArrayLayout::<3>::new(&[16, 1, 4], &[16, 768, 4], 0);
    // 尝试从第 0 个维度开始合并 2 个维度
    let merged_layout = layout.merge_be(0, 2).unwrap();
    // 断言合并后的形状符合预期
    assert_eq!(merged_layout.shape(), &[16, 4]);
    // 断言合并后的步长符合预期
    assert_eq!(merged_layout.strides(), &[16, 4]);
    // 断言合并后的偏移量符合预期
    assert_eq!(merged_layout.offset(), 0);
}

/// 测试 merge_le 方法的示例用法
#[test]
fn test_merge_le_example() {
    // 创建一个三维数组布局
    let layout = ArrayLayout::<3>::new(&[4, 3, 2], &[1, 4, 12], 0);
    // 从第 0 个维度开始，合并 3 个维度
    let merged_layout = layout.merge_le(0, 3).unwrap();

    // 验证合并后的形状、步长和偏移量
    assert_eq!(merged_layout.shape(), &[24]);
    assert_eq!(merged_layout.strides(), &[1]);
    assert_eq!(merged_layout.offset(), 0);
}

/// 测试合并宽度为 0 时的合并操作
#[test]
fn test_merge_len_zero(){
    // 创建一个三维数组布局
    let layout = ArrayLayout::<3>::new(&[4, 3, 2], &[1, 4, 12], 0);
    // 从第 0 个维度开始，合并 0 个维度
    let merged_layout = layout.merge_le(0, 0).unwrap();

    // 验证合并后的形状、步长和偏移量
    assert_eq!(merged_layout.shape(), &[4, 3, 2]);
    assert_eq!(merged_layout.strides(), &[1, 4, 12]);
    assert_eq!(merged_layout.offset(), 0);
}

/// 测试部分合并操作
#[test]
fn test_partial_merge() {
    // 创建一个四维数组布局
    let layout = ArrayLayout::<4>::new(&[2, 3, 4, 5], &[60, 20, 5, 1], 0);
    // 从第 1 个维度开始，合并 2 个维度
    let merged_layout = layout.merge_be(1, 2).unwrap();

    // 验证合并后的形状、步长和偏移量
    assert_eq!(merged_layout.shape(), &[2, 12, 5]);
    assert_eq!(merged_layout.strides(), &[60, 5, 1]);
    assert_eq!(merged_layout.offset(), 0);
}

/// 测试 merge_free 方法的示例用法
#[test]
fn test_merge_free_example() {
    // 创建一个三维数组布局
    let layout = ArrayLayout::<3>::new(&[3, 2, 4], &[4, 12, 1], 0);
    // 从第 0 个维度开始，合并 3 个维度
    let merged_layout = layout.merge_free(0, 3).unwrap();

    // 验证合并后的形状、步长和偏移量
    assert_eq!(merged_layout.shape(), &[24]);
    assert_eq!(merged_layout.strides(), &[1]);
    assert_eq!(merged_layout.offset(), 0);
}