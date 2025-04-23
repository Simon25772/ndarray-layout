// 引入 crate 中的 ArrayLayout 结构体
use crate::ArrayLayout;
// 引入标准库中的 fmt 模块，用于格式化输出
use std::fmt;

/// 为 ArrayLayout 结构体实现格式化相关方法
impl<const N: usize> ArrayLayout<N> {
    /// 高维数组格式化。
    ///
    /// 该函数根据数组布局信息，将数组元素按照不同维度进行格式化输出。
    ///
    /// # Safety
    ///
    /// 这个函数从对裸指针解引用以获得要格式化的数组元素。
    /// 调用者需要确保指针 `ptr` 有效，并且指向的内存区域包含足够的元素，
    /// 同时偏移量计算不会导致越界访问。
    ///
    /// # 参数
    /// - `f`: 用于格式化输出的 `fmt::Formatter` 引用。
    /// - `ptr`: 指向要格式化的数组元素的裸指针。
    ///
    /// # 返回值
    /// 如果格式化成功，返回 `Ok(())`；否则返回 `fmt::Error`。
    pub unsafe fn write_array<T: fmt::Display + Copy>(
        &self,
        f: &mut fmt::Formatter,
        ptr: *const T,
    ) -> fmt::Result {
        // 根据数组的维度数量进行不同的格式化处理
        match self.ndim() {
            // 处理 0 维数组
            0 => {
                // 写入格式化后的数组信息，从指针偏移处读取元素
                write!(f, "array<> = [{}]", unsafe {
                    ptr.byte_offset(self.offset()).read_unaligned()
                })
            }
            // 处理 1 维数组
            1 => {
                // 解构出数组的形状和步长
                let &[n] = self.shape() else { unreachable!() };
                let &[s] = self.strides() else { unreachable!() };

                // 写入数组标题
                writeln!(f, "array<{n}>[")?;
                // 计算指针偏移
                let ptr = unsafe { ptr.byte_offset(self.offset()) };
                // 遍历数组元素并写入格式化后的信息
                for i in 0..n as isize {
                    writeln!(f, "    {}", unsafe {
                        ptr.byte_offset(i * s).read_unaligned()
                    })?
                }
                // 写入数组结束符
                writeln!(f, "]")?;
                Ok(())
            }
            // 处理多维数组
            _ => {
                // 生成数组标题
                let mut title = "array<".to_string();
                for d in self.shape() {
                    title.push_str(&format!("{d}x"))
                }
                // 移除标题末尾多余的 'x'
                assert_eq!(title.pop(), Some('x'));
                title.push('>');

                // 创建一个栈用于存储索引信息
                let mut stack = Vec::with_capacity(self.ndim() - 2);
                // 递归调用 write_recursive 方法进行格式化
                self.write_recursive(f, ptr, &title, &mut stack)
            }
        }
    }

    /// 递归地格式化多维数组。
    ///
    /// 该函数通过递归的方式处理多维数组的不同维度，将数组元素格式化输出。
    ///
    /// # 参数
    /// - `f`: 用于格式化输出的 `fmt::Formatter` 引用。
    /// - `ptr`: 指向要格式化的数组元素的裸指针。
    /// - `title`: 数组的标题字符串。
    /// - `indices`: 用于存储当前维度索引的可变向量。
    ///
    /// # 返回值
    /// 如果格式化成功，返回 `Ok(())`；否则返回 `fmt::Error`。
    fn write_recursive<T: fmt::Display>(
        &self,
        f: &mut fmt::Formatter,
        ptr: *const T,
        title: &str,
        indices: &mut Vec<usize>,
    ) -> fmt::Result {
        // 根据数组的形状进行不同的格式化处理
        match *self.shape() {
            // 空形状或单元素形状不应该出现，触发 unreachable! 宏
            [] | [_] => unreachable!(),
            // 处理 2 维数组
            [rows, cols] => {
                // 写入数组标题和索引信息
                write!(f, "{title}[")?;
                for i in indices {
                    write!(f, "{i}, ")?
                }
                writeln!(f, "..]")?;

                // 解构出数组的行步长和列步长
                let &[rs, cs] = self.strides() else {
                    unreachable!()
                };

                // 计算指针偏移
                let ptr = unsafe { ptr.byte_offset(self.offset()) };
                // 遍历二维数组的行和列，写入格式化后的元素信息
                for r in 0..rows as isize {
                    for c in 0..cols as isize {
                        write!(f, "{} ", unsafe {
                            ptr.byte_offset(r * rs + c * cs).read_unaligned()
                        })?
                    }
                    writeln!(f)?
                }
            }
            // 处理多维数组的批量维度
            [batch, ..] => {
                // 遍历批量维度
                for i in 0..batch {
                    // 将当前索引压入栈中
                    indices.push(i);
                    // 递归调用 write_recursive 方法处理下一个维度
                    self.index(0, i).write_recursive(f, ptr, title, indices)?;
                    // 从栈中弹出当前索引
                    indices.pop();
                }
            }
        }
        Ok(())
    }
}

/// 测试格式化功能的测试用例
#[test]
fn test() {
    // 定义测试数据
    const DATA: &[u8] = &[1, 2, 3, 4, 5, 6, 7, 8, 9, 0];

    /// 定义一个包装结构体 Tensor，包含 ArrayLayout
    struct Tensor(ArrayLayout<4>);

    /// 为 Tensor 结构体实现 fmt::Display trait，用于格式化输出
    impl fmt::Display for Tensor {
        /// 实现 fmt 方法，调用 ArrayLayout 的 write_array 方法进行格式化
        fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
            unsafe { self.0.write_array(f, DATA.as_ptr()) }
        }
    }

    // 创建一个 1 维数组布局的 Tensor 实例并打印
    let tensor = Tensor(ArrayLayout::<4>::new_contiguous(
        &[DATA.len()],
        crate::Endian::BigEndian,
        1,
    ));
    println!("{}", tensor);

    // 对数组布局进行平铺和广播操作后创建新的 Tensor 实例并打印
    let tensor = Tensor(tensor.0.tile_be(0, &[1, DATA.len()]).broadcast(0, 6));
    println!("{}", tensor);

    // 对数组布局进行多次平铺操作后创建新的 Tensor 实例并打印
    let tensor = Tensor(tensor.0.tile_be(0, &[2, 3]).tile_be(2, &[5, 2]));
    println!("{}", tensor);

    // 创建一个 0 维数组布局的 Tensor 实例并打印
    let tensor = Tensor(ArrayLayout::<4>::with_ndim(0));
    println!("{}", tensor);
}