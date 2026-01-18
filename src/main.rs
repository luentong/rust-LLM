use anyhow::Result;
use candle_core::{Device, Tensor};
use std::time::Instant;

fn main() -> Result<()> {
    println!("🚀 Rust LLM Simple");
    println!("📦 使用 candle-core 0.4.0");
    
    // 测试基本功能
    let device = Device::Cpu;
    println!("📱 设备: {:?}", device);
    
    // 1. 创建简单张量
    println!("\n🧪 测试 1: 简单张量");
    let tensor = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], (2, 2), &device)?;
    println!("   创建成功: {:?}", tensor.shape());
    
    // 2. 随机张量
    println!("\n🧪 测试 2: 随机张量");
    let random_tensor = Tensor::randn(0.0, 1.0, (3, 3), &device)?;
    println!("   随机张量形状: {:?}", random_tensor.shape());
    
    // 3. 矩阵乘法
    println!("\n🧪 测试 3: 矩阵乘法");
    let a = Tensor::from_vec(
        vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        (2, 3),
        &device
    )?;
    
    let b = Tensor::from_vec(
        vec![7.0, 8.0, 9.0, 10.0, 11.0, 12.0],
        (3, 2),
        &device
    )?;
    
    let c = a.matmul(&b)?;
    println!("   矩阵乘法成功!");
    println!("   结果形状: {:?}", c.shape());
    
    println!("\n🎉 所有测试通过!");
    println!("�� candle-core 0.4.0 可以正常工作");
    
    Ok(())
}
