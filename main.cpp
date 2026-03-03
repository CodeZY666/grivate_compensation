/*
function:piper_ros机械臂实现重力补偿效果，效果近似piper_ros的示教模式
data:2026/3/3
Author:zy
*/
#!/usr/bin/env python3
# -*-coding:utf8-*-

import time
import numpy as np
import mujoco
import os
from piper_sdk import *

class PiperMITGravityCompensation:
    def __init__(self):
        """使用JointMitCtrl实现重力补偿"""
        self.model_path = r"/home/robot/piper_ws/src/piper_ros/src/piper_description/mujoco_model/piper_no_gripper_description.xml"
        
        self.piper = C_PiperInterface_V2("can0")
        self.connect_robot()
        self.init_mujoco()
        
        self.arm_joint_num = 6
        self.control_freq = 200
        self.dt = 1.0 / self.control_freq
        
        self.max_torque = np.array([15.0, 15.0, 12.0, 8.0, 6.0, 4.0])
        self.prev_torque = np.zeros(self.arm_joint_num)
        self.alpha = 0.2
        
        # 初始化速度计算相关变量
        self.prev_positions = None
        self.prev_time = None
        
        self.target_positions = None
        self.loop_counter = 0
        self.position_update_rate = 100
        self.use_position_hold = True
        
        print("✅ 使用JointMitCtrl的重力补偿控制器初始化完成")
    
    def connect_robot(self):
        """连接机械臂（不激活MIT模式）"""
        try:
            print("正在连接Piper机械臂...")
            self.piper.ConnectPort()
            
            while not self.piper.EnablePiper():
                time.sleep(0.01)
            print("✅ 机械臂使能成功")
            
            print("✅ 连接完成，等待MIT模式激活")
            
        except Exception as e:
            print(f"❌ 连接错误: {e}")
            raise
    
    def init_mujoco(self):
        """初始化MuJoCo物理引擎"""
        try:
            if not os.path.exists(self.model_path):
                raise FileNotFoundError(f"模型文件不存在: {self.model_path}")
            
            print(f"加载MuJoCo模型: {self.model_path}")
            
            model_dir = os.path.dirname(self.model_path)
            original_cwd = os.getcwd()
            os.chdir(model_dir)
            
            try:
                self.model = mujoco.MjModel.from_xml_path("piper_no_gripper_description.xml")
                self.data = mujoco.MjData(self.model)
            finally:
                os.chdir(original_cwd)
            
            print(f"✅ MuJoCo模型加载成功: {self.model.nq} 个关节")
            
            # 配置重力 - 水平桌面安装
            self.configure_horizontal_gravity()
                
        except Exception as e:
            print(f"❌ MuJoCo模型加载失败: {e}")
            raise
    
    def configure_side_mount_gravity(self):
        """
            配置侧面(绕X轴旋转75度)安装的重力
        
        """
        # 侧面重力
        g_side = np.arry([0.0, 0.0, -9.81])
        
        # 旋转方向
        angel = 75.0
        
        # 重力旋转向量为
        R = np.arry([[1, 0, 0],
                     [0, np.cos(angel), -np.sin(angel)],
                     [0, np.sin(angel), np.cos(angel)]
                     ])

        # 计算侧面安装的重力向量
        g_side_mount = R.T @ g_side
        
        # 应用到模型
        self.model.opt.gravity[:] = g_side_mount
        
        # 验证
        self._verify_gravity_config()
        
        return g_side_mount
        
    def configure_horizontal_gravity(self):
        # 标准竖直重力（无旋转）
        g_horizontal = np.array([0.0, 0.0, -9.81])
        
        print(f"  重力向量: {g_horizontal}")
        print(f"  大小: {np.linalg.norm(g_horizontal):.4f} m/s²")
        
        # 应用到模型
        self.model.opt.gravity[:] = g_horizontal
        
        # 验证
        self._verify_gravity_config()
        
        return g_horizontal
    
    def _verify_gravity_config(self):
        
        # 初始姿态的重力力矩
        self.data.qpos[:6] = 0
        self.data.qvel[:] = 0
        self.data.qacc[:] = 0
        mujoco.mj_forward(self.model, self.data)
        tau_g_init = self.data.qfrc_bias[:6].copy()
        
        print(f"  初始姿态 (所有关节=0°) 的重力力矩:")
        for i in range(6):
            print(f"    关节{i+1}: {tau_g_init[i]:+.6f} Nm")
        
        print(f"\n✅ 验证完成\n")
    
    def get_joint_states(self):
        """获取前6个关节的状态"""
        try:
            arm_joint_msg = self.piper.GetArmJointMsgs()
            
            joint_encoders = [
                arm_joint_msg.joint_state.joint_1,
                arm_joint_msg.joint_state.joint_2,
                arm_joint_msg.joint_state.joint_3,
                arm_joint_msg.joint_state.joint_4,
                arm_joint_msg.joint_state.joint_5,
                arm_joint_msg.joint_state.joint_6
            ]
            
            arm_positions = self.encoder_to_radians(joint_encoders)
            arm_velocities = self.calculate_joint_velocities(arm_positions)
            
            return arm_positions, arm_velocities
            
        except Exception as e:
            print(f"❌ 获取关节状态失败: {e}")
            return None, None
    
    def encoder_to_radians(self, encoder_values):
        """将编码器值转换为弧度"""
        try:
            if not isinstance(encoder_values, (list, tuple)) or len(encoder_values) != 6:
                print(f"❌ 编码器数据格式错误: {encoder_values}")
                return np.zeros(6)
            
            zero_offsets = [0, 0, 0, 0, 0, 0]
            
            radians = []
            for i, encoder_count in enumerate(encoder_values):
                try:
                    adjusted_count = encoder_count - zero_offsets[i]
                    angle_deg = adjusted_count * 0.001
                    angle_rad = np.radians(angle_deg)
                    radians.append(angle_rad)
                except (TypeError, ValueError) as e:
                    print(f"❌ 关节{i+1}编码器值错误: {encoder_count}")
                    radians.append(0.0)
            
            return np.array(radians)
            
        except Exception as e:
            print(f"❌ 编码器转换失败: {e}")
            return np.zeros(6)
    
    def calculate_joint_velocities(self, current_positions):
        """通过位置差分计算关节速度"""
        current_time = time.time()
        
        if self.prev_positions is None or self.prev_time is None:
            self.prev_positions = current_positions.copy()
            self.prev_time = current_time
            return np.zeros(self.arm_joint_num)
        
        dt = current_time - self.prev_time
        if dt <= 0:
            return np.zeros(self.arm_joint_num)
        
        velocities = (current_positions - self.prev_positions) / dt
        
        self.prev_positions = current_positions.copy()
        self.prev_time = current_time
        
        return velocities
    
    def sync_to_mujoco(self, arm_positions, arm_velocities=None):
        """将机械臂状态同步到MuJoCo"""
        if arm_velocities is None:
            arm_velocities = np.zeros(self.arm_joint_num)
        
        self.data.qpos[:self.arm_joint_num] = arm_positions
        self.data.qvel[:self.arm_joint_num] = arm_velocities
        
        if self.model.nq > self.arm_joint_num:
            self.data.qpos[self.arm_joint_num:] = 0.0
            self.data.qvel[self.arm_joint_num:] = 0.0
        
        self.data.qacc[:] = 0.0
        mujoco.mj_forward(self.model, self.data)
    
    def compute_gravity_compensation(self):
        # 保存原始状态
        original_vel = self.data.qvel.copy()
        original_acc = self.data.qacc.copy()
        
        # 设置零速度、零加速度（计算静态重力）
        self.data.qvel[:] = 0.0
        self.data.qacc[:] = 0.0
        
        # 计算重力、科里奥利力等
        mujoco.mj_forward(self.model, self.data)
        
        # 提取重力补偿力矩
        gravity_torque = self.data.qfrc_bias[:self.arm_joint_num].copy()
        
        # 恢复原始状态
        self.data.qvel[:] = original_vel
        self.data.qacc[:] = original_acc
        
        return gravity_torque
    
    def apply_safety_limits(self, torque):
        """应用安全限制和滤波"""
        limited_torque = np.clip(torque, -self.max_torque, self.max_torque)
        api_safe_torque = np.clip(limited_torque, -18.0, 18.0)
        
        filtered_torque = (self.alpha * api_safe_torque + 
                          (1 - self.alpha) * self.prev_torque)
        self.prev_torque = filtered_torque.copy()
        
        return filtered_torque
    
    def test_horizontal_gravity_compensation(self, duration=50, kp=0.1, kd=0.08):
        """
        水平桌面安装的重力补偿测试
        """
        try:
            # 1. 激活MIT模式
            print("激活MIT模式...")
            self.piper.MotionCtrl_2(0x01, 0x04, 0, 0xAD)
            time.sleep(0.5)
            print("MIT模式激活成功")
            
            # 2. 读取初始位置
            print("读取初始位置...")
            initial_pos, _ = self.get_joint_states()
            if initial_pos is None:
                print("❌ 无法读取关节状态")
                return False
            
            # 3. 重置速度计算
            self.prev_positions = None
            self.prev_time = None
            
            # 4. 验证重力补偿计算
            print("\n验证重力补偿计算...")
            self.sync_to_mujoco(initial_pos, np.zeros(6))
            initial_gravity = self.compute_gravity_compensation()
            
            input("按回车开始测试...")
            
            loop_count = 0
            start_time = time.time()
            max_drift = np.zeros(6)
            
            # 5. 主控制循环
            print(f"\n运行中... (按 Ctrl+C 停止)")
            while True:
                loop_start = time.perf_counter()
                
                # 读取当前状态
                positions, velocities = self.get_joint_states()
                if positions is None:
                    continue
                
                # 同步到MuJoCo
                self.sync_to_mujoco(positions, velocities)
                
                # 计算重力力矩
                gravity_torque = self.compute_gravity_compensation()

                # 定义列表存储力矩
                tau_g_list = []
                for i in range(6):
                    tau_g = gravity_torque[i]
                    # 根据不同的关节设置不同的力矩限制
                    if i == 0:
                        tau_g = np.clip(tau_g, -0.5, 0.5)
                    elif i == 1:
                        # tau_g = np.clip(tau_g, -3.0, 0.5)  # 力矩已精调
                        tau_g = tau_g * 0.3
                        # print("第二关节力矩:", tau_g)
                    elif i == 2:
                        # tau_g = np.clip(tau_g, -1.0, 1.0)
                        if(tau_g < 0):
                            tau_g = tau_g * 0.3
                        else:
                            tau_g = tau_g * 0.3
                        # print("第三关节力矩:", tau_g)
                    elif i == 3:
                        tau_g = np.clip(tau_g, -0.9, 1.0) 
                    elif i == 4:
                        if(tau_g < 0):
                            tau_g = tau_g * 1.7
                        else:
                            tau_g = tau_g * 1.7  
                    elif i == 5:
                        tau_g = np.clip(tau_g, -1.0, 1.0)
                    tau_g_list.append(tau_g)
                
                # for j in range(1, 7):
                #     print(f"关节{j}的位置是：{positions[j-1]},关节{j}的力矩是：{tau_g_list[j-1]}")
                
                # 发送MIT控制命令 - 所有关节同时补偿 
                try:    
                    # 关节1已经调试完成
                    self.piper.JointMitCtrl(
                        1,
                        pos_ref=float(positions[0]),      
                        vel_ref=0.0,
                        kp=0.03,
                        kd=0.01,
                        t_ref=float(tau_g_list[0])       
                        )                     
                    
                    # 关节2已经调试完成
                    self.piper.JointMitCtrl(
                        2,
                        pos_ref=float(positions[1]),      
                        vel_ref=0.0,
                        kp=0.05,
                        kd=0.05,
                        t_ref=float(tau_g_list[1])       
                        )       
                    
                    # 关节3已经调试完成
                    self.piper.JointMitCtrl(
                        3,
                        pos_ref=float(positions[2]),      
                        vel_ref=0.0,
                        kp=0.03,
                        kd=0.01,
                        t_ref=float(tau_g_list[2])       
                        )       

                    self.piper.JointMitCtrl(
                        4,
                        pos_ref=float(positions[3]),      
                        vel_ref=0.0,
                        kp=0.1,
                        kd=0.08,
                        t_ref=float(tau_g_list[3] * 1.5)       
                        )                      
                    
                    # 关节5已经调试完成
                    self.piper.JointMitCtrl(
                        5,
                        pos_ref=float(positions[4]),      
                        vel_ref=0.0,
                        kp=0.1,
                        kd=0.08,
                        t_ref=float(tau_g_list[4])       
                        )       

                    self.piper.JointMitCtrl(
                        6,
                        pos_ref=float(positions[5]),      
                        vel_ref=0.0,
                        kp=0.1,
                        kd=0.08,
                        t_ref=float(tau_g_list[5])      
                        )                     
                    
                except Exception as e:
                    print(f"MIT控制命令发送失败: {e}")
                    continue
                
                # 记录最大漂移
                drift = np.abs(positions - initial_pos)
                max_drift = np.maximum(max_drift, drift)
                
                # 定期打印状态
                loop_count += 1
                if loop_count % 200 == 0:  # 每1秒打印
                    elapsed = time.time() - start_time
                    drift_deg = np.degrees(positions - initial_pos)
                    print(f"[{elapsed:5.1f}s] 位置漂移(°): {drift_deg}")
                    print(f"           重力力矩(Nm): {gravity_torque}")
                
                # 精确时间控制 (200Hz)
                loop_time = time.perf_counter() - loop_start
                sleep_time = 0.005 - loop_time
                if sleep_time > 0:
                    time.sleep(sleep_time)
                  
        except KeyboardInterrupt:
            print("\n✅ 用户停止测试")
        
        except Exception as e:
            print(f"\n❌ 测试过程中出错: {e}")
            import traceback
            traceback.print_exc()
            return False
        
        finally:
            # 安全停止流程
            print("\n正在安全停止...")
            
            try:
                # 用当前位置平稳停止
                final_pos, _ = self.get_joint_states()
                if final_pos is not None:
                    print("平稳停止中...")
                    for _ in range(100):  # 0.5秒
                        for j in range(1, 7):
                            self.piper.JointMitCtrl(j, float(final_pos[j-1]), 0.0, 0.0, 0.2, 0.0)
                        time.sleep(0.005)
                
                # 归零控制
                print("归零控制...")
                for j in range(1, 7):
                    self.piper.JointMitCtrl(j, 0.0, 0.0, 0.0, 0.0, 0.0)
                time.sleep(0.1)
                
                # 显示测试结果
                if final_pos is not None:
                    final_drift = np.degrees(final_pos - initial_pos)
                    max_total_drift = np.max(np.abs(final_drift))
                    
                    # 判断测试结果
                    if max_total_drift < 2.0:
                        return True
                    elif max_total_drift < 10.0:
                        return None
                    else:
                        return False
                    
            except Exception as e:
                print(f"❌ 停止过程中出错: {e}")
            
            print("✅ 已安全停止\n")
    
    def test_with_different_params(self):
        """
        【新增】用不同的Kp/Kd参数测试
        """
        print("\n" + "="*80)
        print("用不同参数进行多次测试")
        print("="*80)
        
        test_params = [
            (0.0, 0.0, "纯前馈补偿（无PID）"),
            (0.1, 0.1, "弱PID + 补偿"),
            (0.5, 0.2, "中等PID + 补偿"),
            (1.0, 0.3, "强PID + 补偿"),
        ]
        
        results = []
        
        for kp, kd, description in test_params:
            print(f"\n测试: {description} (kp={kp}, kd={kd})")
            
            result = self.test_horizontal_gravity_compensation(duration=5, kp=kp, kd=kd)
            results.append((description, result))
            
            time.sleep(1)  # 测试间隔
        
        print("\n" + "="*80)
        print("所有测试结果汇总")
        print("="*80)
        for desc, result in results:
            status = "✅ PASS" if result is True else ("⚠️  PARTIAL" if result is None else "❌ FAIL")
            print(f"  {status}: {desc}")
    
    def verify_gravity_with_current_pose(self):
        """验证当前姿态的重力补偿"""
        print("=== 验证当前姿态的重力补偿 ===")
        
        positions, _ = self.get_joint_states()
        if positions is None:
            print("❌ 无法读取关节状态\n")
            return
        
        self.sync_to_mujoco(positions, np.zeros(6))
        tau_g = self.compute_gravity_compensation()
        
        print(f"当前姿态:")
        print(f"  关节角度(度): {np.degrees(positions)}")
        print(f"  重力力矩(Nm): {tau_g}\n")
    
    def manual_position_update_mode(self):
        """
        原有的单关节测试 - 保持不变
        """
        print("\n" + "="*70)
        print("="*70 + "\n")
        
        print("测试说明:")
        print("  • 只测试关节2的重力补偿")
        print("  • 其他关节保持当前位置")
        print("  • 手动移动关节2，松手后应该停住")
        print("  • 按 Ctrl+C 停止\n")
        
        try:
            # 1. 激活MIT模式
            print("激活MIT模式...")
            self.piper.MotionCtrl_2(0x01, 0x04, 0, 0xAD)
            time.sleep(0.5)
            print("✅ MIT模式激活成功")
            
            # 2. 读取初始位置
            print("读取初始位置...")
            initial_pos, _ = self.get_joint_states()
            if initial_pos is None:
                print("❌ 无法读取关节状态")
                return
            
            print(f"✅ 初始关节2位置: {np.degrees(initial_pos[1]):+.2f}°")
            
            # 3. 重置速度计算
            self.prev_positions = None
            self.prev_time = None
            
            # 4. 验证重力补偿计算
            self.sync_to_mujoco(initial_pos, np.zeros(6))
            initial_gravity = self.compute_gravity_compensation()
            
            input("按回车开始测试...")
            
            loop_count = 0
            start_time = time.time()
            
            # 5. 主控制循环
            while True:
                loop_start = time.perf_counter()
                
                # 读取当前状态
                positions, velocities = self.get_joint_states()
                if positions is None:
                    print("跳过此次循环：无法读取关节状态")
                    continue
                
                self.sync_to_mujoco(positions, velocities)
                gravity_torque = self.compute_gravity_compensation()
                tau_g0 = gravity_torque[0]
                tau_g0 = np.clip(tau_g0, -2.5, 2.5)

                try:
                    self.piper.JointMitCtrl(
                        1,
                        float(positions[0]),
                        0.0,
                        0.0,
                        0.0,
                        float(tau_g0)
                    )
                    print("第1关节:",positions[0])
                    print("力矩:",tau_g0)

                    for j in [1, 3, 4, 5, 6]:
                        self.piper.JointMitCtrl(
                            j,
                            float(positions[j-1]),
                            0.0,
                            0.001,    
                            0.001,    
                            0.0
                        )
                    
                except Exception as e:
                    print(f"MIT控制命令发送失败: {e}")
                    continue
                
                loop_count += 1
                if loop_count % 100 == 0:
                    elapsed = time.time() - start_time
                    j2_deg = np.degrees(positions[1])
                    j2_vel = np.degrees(velocities[1])
                    drift = j2_deg - np.degrees(initial_pos[1])
                
                loop_time = time.perf_counter() - loop_start
                sleep_time = 0.005 - loop_time
                if sleep_time > 0:
                    time.sleep(sleep_time)
                  
        except KeyboardInterrupt:
            print("\n✅ 用户停止测试")
        
        except Exception as e:
            print(f"\n❌ 测试过程中出错: {e}")
            import traceback
            traceback.print_exc()
        
        finally:
            print("\n正在安全停止...")
            
            try:
                final_pos, _ = self.get_joint_states()
                if final_pos is not None:
                    print("平稳停止中...")
                    for _ in range(100):
                        for j in range(1, 7):
                            self.piper.JointMitCtrl(j, float(final_pos[j-1]), 0.0, 0.0, 0.2, 0.0)
                        time.sleep(0.005)
                
                print("归零控制...")
                for j in range(1, 7):
                    self.piper.JointMitCtrl(j, 0.0, 0.0, 0.0, 0.0, 0.0)
                time.sleep(0.1)
                
                if final_pos is not None and initial_pos is not None:
                    final_j2_deg = np.degrees(final_pos[1])
                    initial_j2_deg = np.degrees(initial_pos[1])
                    total_drift = final_j2_deg - initial_j2_deg
                    
            except Exception as e:
                print(f"❌ 停止过程中出错: {e}")
            
            print("✅ 已安全停止\n")
    
    def emergency_stop(self):
        """紧急停止"""
        print("执行紧急停止...")
        
        try:
            for joint_id in range(1, self.arm_joint_num + 1):
                self.piper.JointMitCtrl(joint_id, 0.0, 0.0, 0.0, 0.0, 0.0)
            
            time.sleep(0.1)
            self.piper.MotionCtrl_2(0x00, 0x00, 0, 0xAD)
            self.piper.DisConnectPort()
            
            print("✅ 紧急停止完成")
            
        except Exception as e:
            print(f"❌ 紧急停止出错: {e}")


def test_single_joint():
    """测试单关节MIT控制"""
    try:
        piper = C_PiperInterface_V2("can0")
        piper.ConnectPort()
        
        while not piper.EnablePiper():
            time.sleep(0.01)
        
        print("测试单关节MIT控制...")
        controller = PiperMITGravityCompensation()
        
        for i in range(100):
            print(f"测试 {i+1}/100")
            
            piper.MotionCtrl_2(0x01, 0x04, 0, 0xAD)
            piper.JointMitCtrl(1, 0.0, 0.0, 0.1, 0.04, 0.1)
            
            positions, _ = controller.get_joint_states()
            time.sleep(0.1)
        
        # 全关节失能
        for i in range(1,7):
            piper.JointMitCtrl(i, 0.0, 0.0, 0.0, 0.0, 0.0) 
            
        time.sleep(0.1)
        piper.MotionCtrl_2(0x00, 0x00, 0, 0xAD)
        piper.DisConnectPort()
        
        print("单关节测试完成")
        
    except Exception as e:
        print(f"单关节测试失败: {e}")

def main():
    print("="*80)
    print("Piper机械臂重力补偿测试 - 水平桌面安装")
    print("="*80)
    print("\n请选择测试模式:")
    print("  1. 水平桌面 - 基础重力补偿测试 (推荐先选这个)")
    print("  2. 水平桌面 - 多参数对比测试")
    print("  3. 单关节重力补偿测试")
    print("  4. 单关节MIT控制测试")
    print("  5. 验证当前姿态的重力补偿")
    print("  6. 退出")
    
    choice = input("\n请选择 (1-6): ").strip()
    
    try:
        if choice == "1":
            controller = PiperMITGravityCompensation()
            controller.test_horizontal_gravity_compensation()
        elif choice == "2":
            controller = PiperMITGravityCompensation()
            controller.test_with_different_params()
        elif choice == "3":
            controller = PiperMITGravityCompensation()
            controller.manual_position_update_mode()
        elif choice == "4":
            test_single_joint()
        elif choice == "5":
            controller = PiperMITGravityCompensation()
            controller.verify_gravity_with_current_pose()
        else:
            print("无效选择")
            
    except Exception as e:
        print(f"程序错误: {e}")
        import traceback
        traceback.print_exc()
    finally:
        print("程序结束")

if __name__ == "__main__":
    main()
