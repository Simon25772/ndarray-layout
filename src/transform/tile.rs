// 引入 crate 中的 ArrayLayout 结构体和 Endian 枚举
use crate::{ArrayLayout, Endian};
// 引入标准库中的 zip 函数，用于同时迭代多个迭代器
use std::iter::zip;

/// 分块变换参数。该结构体用于存储分块变换所需的参数。
#[derive(Clone, PartialEq, Eq, Debug)]
pub struct TileArg<'a> {
    /// 分块操作要应用的轴。轴的索引从 0 开始。
    pub axis: usize,
    /// 分块的顺序。大端序或小端序决定了分块后维度在形状中的排列顺序。
    pub endian: Endian,
    /// 分块的大小数组。每个元素表示对应分块的大小。
    pub tiles: &'a [usize],
}

/// 为 ArrayLayout 结构体实现分块变换相关方法
impl<const N: usize> ArrayLayout<N> {
    /// 大端分块变换。将单个维度划分为多个分块，大端分块使得分块后范围更大的维度在形状中更靠前的位置。
    ///
    /// # 示例
    /// ```rust
    /// # use ndarray_layout::ArrayLayout;
    /// let layout = ArrayLayout::<3>::new(&[2, 3, 6], &[18, 6, 1], 0).tile_be(2, &[2, 3]);
    /// assert_eq!(layout.shape(), &[2, 3, 2, 3]);
    /// assert_eq!(layout.strides(), &[18, 6, 3, 1]);
    /// assert_eq!(layout.offset(), 0);
    /// ```
    ///
    /// # 参数
    /// - `axis`: 要进行分块的轴的索引。
    /// - `tiles`: 分块大小的数组。
    ///
    /// # 返回值
    /// 返回一个新的 `ArrayLayout` 实例，其维度已根据大端分块规则进行变换。
    #[inline]
    pub fn tile_be(&self, axis: usize, tiles: &[usize]) -> Self {
        // 调用 tile_many 方法，传入大端序的分块参数
        self.tile_many(&[TileArg {
            axis,
            endian: Endian::BigEndian,
            tiles,
        }])
    }

    /// 小端分块变换。将单个维度划分为多个分块，小端分块使得分块后范围更小的维度在形状中更靠前的位置。
    ///
    /// # 示例
    /// ```rust
    /// # use ndarray_layout::ArrayLayout;
    /// let layout = ArrayLayout::<3>::new(&[2, 3, 6], &[18, 6, 1], 0).tile_le(2, &[2, 3]);
    /// assert_eq!(layout.shape(), &[2, 3, 2, 3]);
    /// assert_eq!(layout.strides(), &[18, 6, 1, 2]);
    /// assert_eq!(layout.offset(), 0);
    /// ```
    ///
    /// # 参数
    /// - `axis`: 要进行分块的轴的索引。
    /// - `tiles`: 分块大小的数组。
    ///
    /// # 返回值
    /// 返回一个新的 `ArrayLayout` 实例，其维度已根据小端分块规则进行变换。
    #[inline]
    pub fn tile_le(&self, axis: usize, tiles: &[usize]) -> Self {
        // 调用 tile_many 方法，传入小端序的分块参数
        self.tile_many(&[TileArg {
            axis,
            endian: Endian::LittleEndian,
            tiles,
        }])
    }

    /// 一次对多个阶进行分块变换。
    ///
    /// # 参数
    /// - `args`: 包含多个 `TileArg` 结构体的切片，每个结构体表示一个轴的分块参数。
    ///
    /// # 返回值
    /// 返回一个新的 `ArrayLayout` 实例，其维度已根据所有分块参数进行变换。
    pub fn tile_many(&self, mut args: &[TileArg]) -> Self {
        // 获取当前布局的内容
        let content = self.content();
        // 获取当前布局的形状
        let shape = content.shape();
        // 同时迭代形状和步长，并获取索引
        let iter = zip(shape, content.strides()).enumerate();

        // 定义一个闭包，用于检查分块参数是否有效
        let check = |&TileArg { axis, tiles, .. }| {
            // 检查指定轴的维度大小是否等于分块大小的乘积
            shape
                .get(axis)
                .filter(|&&d| d == tiles.iter().product())
                .is_some()
        };

        // 初始化新布局的维度数量和上一个处理的轴的索引
        let (mut new, mut last_axis) = match args {
            [first, ..] => {
                // 检查第一个分块参数是否有效
                assert!(check(first));
                // 新布局的维度数量初始化为第一个分块参数的分块数量
                (first.tiles.len(), first.axis)
            }
            [..] => return self.clone(), // 如果没有分块参数，直接克隆当前布局
        };
        // 遍历剩余的分块参数
        for arg in &args[1..] {
            // 检查分块参数是否有效
            assert!(check(arg));
            // 确保当前轴的索引大于上一个轴的索引
            assert!(arg.axis > last_axis);
            // 累加新布局的维度数量
            new += arg.tiles.len();
            // 更新上一个处理的轴的索引
            last_axis = arg.axis;
        }

        // 创建一个新的 ArrayLayout 实例，其维度数量为当前布局的维度数量加上新维度数量减去分块参数的数量
        let mut ans = Self::with_ndim(self.ndim + new - args.len());

        // 获取新布局内容的可变引用
        let mut content = ans.content_mut();
        // 将新布局的偏移量设置为当前布局的偏移量
        content.set_offset(self.offset());
        // 初始化新布局的索引
        let mut j = 0;
        // 定义一个闭包，用于设置新布局的形状和步长
        let mut push = |t, s| {
            content.set_shape(j, t);
            content.set_stride(j, s);
            j += 1;
        };

        // 遍历当前布局的形状和步长
        for (i, (&d, &s)) in iter {
            match *args {
                [
                    TileArg {
                        axis,
                        endian,
                        tiles,
                    },
                    ref tail @ ..,
                ] if axis == i => {
                    // 如果当前轴与分块参数的轴匹配
                    match endian {
                        Endian::BigEndian => {
                            // 大端分块规则
                            // tile   : [a,         b    , c]
                            // strides: [s * c * b, s * c, s]
                            let mut s = s * d as isize;
                            for &t in tiles {
                                s /= t as isize;
                                push(t, s);
                            }
                        }
                        Endian::LittleEndian => {
                            // 小端分块规则
                            // tile   : [a, b    , c        ]
                            // strides: [s, s * a, s * a * b]
                            let mut s = s;
                            for &t in tiles {
                                push(t, s);
                                s *= t as isize;
                            }
                        }
                    }
                    // 处理完当前分块参数后，更新剩余的分块参数
                    args = tail;
                }
                [..] => push(d, s), // 如果当前轴没有分块参数，直接设置形状和步长
            }
        }
        // 返回新的布局
        ans
    }
}

/// 测试大端分块变换的正确性
#[test]
fn test_tile_be() {
    let layout = ArrayLayout::<3>::new(&[2, 3, 6], &[18, 6, 1], 0).tile_be(2, &[2, 3]);
    assert_eq!(layout.shape(), &[2, 3, 2, 3]);
    assert_eq!(layout.strides(), &[18, 6, 3, 1]);
    assert_eq!(layout.offset(), 0);
}

/// 测试小端分块变换的正确性
#[test]
fn test_tile_le() {
    let layout = ArrayLayout::<3>::new(&[2, 3, 6], &[18, 6, 1], 0).tile_le(2, &[2, 3]);
    assert_eq!(layout.shape(), &[2, 3, 2, 3]);
    assert_eq!(layout.strides(), &[18, 6, 1, 2]);
    assert_eq!(layout.offset(), 0);
}

/// 测试无分块参数时的行为
#[test]
fn test_empty_tile() {
    let layout = ArrayLayout::<3>::new(&[2, 3, 6], &[18, 6, 1], 0).tile_many(&[]);
    assert_eq!(layout.shape(), &[2, 3, 6]);
    assert_eq!(layout.strides(), &[18, 6, 1]);
    assert_eq!(layout.offset(), 0);
}

/// 测试多个分块参数时的行为
#[test]
fn test_multiple_tiles(){
    let layout = ArrayLayout::<3>::new(&[2, 3, 6], &[18, 6, 1], 0).tile_many(&[
        TileArg {
            axis: 0,
            endian: Endian::BigEndian,
            tiles: &[2, 1],
        },
        TileArg {
            axis: 2,
            endian: Endian::BigEndian,
            tiles: &[2, 3],
        }
    ]);
    assert_eq!(layout.shape(), &[2, 1, 3, 2, 3]);
    assert_eq!(layout.strides(), &[18, 18, 6, 3, 1]);
    assert_eq!(layout.offset(), 0);
}