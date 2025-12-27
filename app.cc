// Copyright 2021 DeepMind Technologies Limited
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "mjpc/app.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <memory>
#include <mutex>
#include <string>
#include <string_view>
#include <thread>
#include <utility>
#include <vector>

#include <absl/flags/flag.h>
#include <mujoco/mujoco.h>
#include <glfw_adapter.h>
#include "mjpc/array_safety.h"
#include "mjpc/agent.h"
#include "mjpc/estimators/estimator.h"
#include "mjpc/simulate.h"  // mjpc fork
#include "mjpc/task.h"
#include "mjpc/threadpool.h"
#include "mjpc/utilities.h"

// 优化：添加命名空间别名，提高可读性
namespace mj = ::mujoco;
namespace mju = ::mujoco::util_mjpc;

ABSL_FLAG(bool, planner_enabled, false,
          "If true, the planner will run on startup");
ABSL_FLAG(float, sim_percent_realtime, 100,
          "The realtime percentage at which the simulation will be launched.");
ABSL_FLAG(bool, estimator_enabled, false,
          "If true, estimator loop will run on startup");
ABSL_FLAG(bool, show_left_ui, true,
          "If true, the left UI (ui0) will be visible on startup");
ABSL_FLAG(bool, show_plot, true,
          "If true, the plots will be visible on startup");
ABSL_FLAG(bool, show_info, true,
          "If true, the infotext panel will be visible on startup");


namespace {
// 优化：使用更具描述性的常量名称
constexpr double kSyncMisalignmentThreshold = 0.1;  // 最大不同步阈值（秒）
constexpr double kSimRefreshFraction = 0.7;         // 用于模拟的刷新时间比例

// 优化：使用智能指针管理资源
static mjModel* g_model = nullptr;
static mjData* g_data = nullptr;
static mjtNum* g_ctrl_noise = nullptr;

using Seconds = std::chrono::duration<double>;

// --------------------------------- 回调函数 ---------------------------------
static std::unique_ptr<mj::Simulate> g_sim;

// 控制器回调
extern "C" void controller(const mjModel* model, mjData* data) {
  // 如果不是主数据，跳过（可能属于其他线程）
  if (data != g_data) {
    return;
  }
  
  // 如果代理启用动作控制
  if (g_sim->agent->action_enabled) {
    g_sim->agent->ActivePlanner().ActionFromPolicy(
        data->ctrl, &g_sim->agent->state.state()[0],
        g_sim->agent->state.time());
  }
  
  // 如果启用控制噪声
  if (!g_sim->agent->allocate_enabled && 
      g_sim->uiloadrequest.load() == 0 &&
      g_sim->ctrl_noise_std > 0.0) {
    for (int j = 0; j < g_sim->m->nu; ++j) {
      data->ctrl[j] += g_ctrl_noise[j];
    }
  }
}

// 传感器回调
extern "C" void sensor(const mjModel* model, mjData* data, int stage) {
  if (stage == mjSTAGE_ACC) {
    if (!g_sim->agent->allocate_enabled && g_sim->uiloadrequest.load() == 0) {
      if (g_sim->agent->IsPlanningModel(model)) {
        // 规划线程和滚动线程不需要同步
        const mjpc::ResidualFn* residual = g_sim->agent->PlanningResidual();
        residual->Residual(model, data, data->sensordata);
      } else {
        // 物理线程和UI线程共享的残差计算
        g_sim->agent->ActiveTask()->Residual(model, data, data->sensordata);
      }
    }
  }
}

//--------------------------------- 模型加载 ---------------------------------

mjModel* LoadModel(const mjpc::Agent* agent, mj::Simulate& sim) {
  mjpc::Agent::LoadModelResult load_result = sim.agent->LoadModel();
  mjModel* new_model = load_result.model.release();
  mju::strcpy_arr(sim.load_error, load_result.error.c_str());

  if (!new_model) {
    std::cerr << "Failed to load model: " << load_result.error << "\n";
    return nullptr;
  }

  // 编译警告：打印并暂停
  if (!load_result.error.empty()) {
    std::cout << "Model compiled with warnings (simulation paused):\n  "
              << load_result.error << "\n";
    sim.run = 0;
  }

  return new_model;
}

// 优化：提取重复的拷贝操作到辅助函数
void CopySimulationStateToEstimator(mjpc::Estimator* estimator, 
                                    const mjData* sim_data,
                                    const mjModel* model) {
  // 拷贝仿真控制信号
  mju_copy(estimator->Data()->ctrl, sim_data->ctrl, model->nu);
  
  // 拷贝仿真传感器数据
  mju_copy(estimator->Data()->sensordata, sim_data->sensordata, 
           model->nsensordata);
  
  // 拷贝仿真时间
  estimator->Data()->time = sim_data->time;
  
  // 拷贝仿真运动捕捉数据
  mju_copy(estimator->Data()->mocap_pos, sim_data->mocap_pos, 
           3 * model->nmocap);
  mju_copy(estimator->Data()->mocap_quat, sim_data->mocap_quat, 
           4 * model->nmocap);
  
  // 拷贝仿真用户数据
  mju_copy(estimator->Data()->userdata, sim_data->userdata, 
           model->nuserdata);
}

// 估计器后台线程
void EstimatorLoop(mj::Simulate& sim) {
  // 运行直到请求退出
  while (!sim.exitrequest.load()) {
    if (sim.uiloadrequest.load() == 0) {
      int active_estimator_index = sim.agent->ActiveEstimatorIndex();
      mjpc::Estimator* estimator = &sim.agent->ActiveEstimator();

      // 估计器更新
      if (active_estimator_index == 0) {
        std::this_thread::yield();
        continue;
      }
      
      // 开始计时
      auto start_time = std::chrono::steady_clock::now();

      // 从GUI设置值
      estimator->SetGUIData();

      // 获取仿真状态（锁定物理线程）
      {
        const std::lock_guard<std::mutex> lock(sim.mtx);
        CopySimulationStateToEstimator(estimator, g_data, g_model);
      }

      // 使用从物理线程拷贝的最新控制信号和传感器数据更新滤波器
      estimator->Update(sim.agent->ctrl.data(), sim.agent->sensor.data());

      // 估计器状态传递给规划器
      double* state = estimator->State();
      sim.agent->state.Set(g_model, state, state + g_model->nq, 
                           state + g_model->nq + g_model->nv,
                           g_data->mocap_pos, g_data->mocap_quat, 
                           g_data->userdata, g_data->time);

      // 等待（微秒）以保持实时性
      const double time_step = estimator->Model()->opt.timestep;
      const double wait_time_us = 1.0e6 * time_step;
      
      while (mjpc::GetDuration(start_time) < wait_time_us) {
        // 忙等待或休眠
      }
    }
  }
}

// 优化：添加辅助函数处理模型重载
bool HandleModelReload(mj::Simulate& sim) {
  // 获取新模型和任务
  sim.filename = sim.agent->GetTaskXmlPath(sim.agent->gui_task_id);

  mjModel* new_model = LoadModel(sim.agent.get(), sim);
  mjData* new_data = nullptr;
  
  if (new_model) {
    new_data = mj_makeData(new_model);
  }
  
  if (!new_data) {
    return false;
  }

  // 初始化代理
  sim.agent->Initialize(new_model);
  sim.agent->plot_enabled = absl::GetFlag(FLAGS_show_plot);
  sim.agent->plan_enabled = absl::GetFlag(FLAGS_planner_enabled);
  sim.agent->Allocate();

  // 设置home关键帧
  int home_id = mj_name2id(new_model, mjOBJ_KEY, "home");
  if (home_id >= 0) {
    mj_resetDataKeyframe(new_model, new_data, home_id);
    sim.agent->Reset(new_data->ctrl);
  } else {
    sim.agent->Reset();
  }
  
  sim.agent->PlotInitialize();
  sim.Load(new_model, new_data, sim.filename, true);
  
  // 更新全局指针
  g_model = new_model;
  g_data = new_data;
  mj_forward(g_model, g_data);

  // 分配控制噪声内存
  free(g_ctrl_noise);
  g_ctrl_noise = static_cast<mjtNum*>(malloc(sizeof(mjtNum) * g_model->nu));
  mju_zero(g_ctrl_noise, g_model->nu);
  
  return true;
}

// 优化：重构模拟步进逻辑
void SimulateStep(mj::Simulate& sim, mjModel* model, mjData* data) {
  // 清空旧扰动，应用新扰动
  mju_zero(data->xfrc_applied, 6 * model->nbody);
  sim.ApplyPosePerturbations(0);  // 仅移动运动捕捉体
  sim.ApplyForcePerturbations();

  // 执行所有步进前任务
  sim.agent->ExecuteAllRunBeforeStepJobs(model, data);
  
  // 执行模拟步进
  mj_step(model, data);
}

// 模拟在后台线程（主线程渲染）
void PhysicsLoop(mj::Simulate& sim) {
  // CPU-仿真同步点
  std::chrono::time_point<mj::Simulate::Clock> sync_cpu_time;
  mjtNum sync_sim_time = 0;

  // 运行直到请求退出
  while (!sim.exitrequest.load()) {
    // 处理拖放加载请求
    if (sim.droploadrequest.load()) {
      // TODO: 在MJPC中实现拖放支持
      sim.droploadrequest.store(false);
    }

    // ----- 任务重新加载 -----
    if (sim.uiloadrequest.load() == 1) {
      if (!HandleModelReload(sim)) {
        std::cerr << "Failed to reload model\n";
      }
      sim.uiloadrequest.fetch_sub(1);
    }

    // GUI重新加载
    if (sim.uiloadrequest.load() == -1) {
      sim.Load(sim.m, sim.d, sim.filename.c_str(), false);
      sim.uiloadrequest.fetch_add(1);
    }
    // -----------------------

    // 休眠1毫秒或让出CPU，让主线程运行
    // 让出CPU会导致忙等待 - 定时更好但耗电
    if (sim.run && sim.busywait) {
      std::this_thread::yield();
    } else {
      std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }

    {
      // 锁定模拟互斥锁
      const std::lock_guard<std::mutex> lock(sim.mtx);

      if (g_model) {  // 仅当模型存在时运行
        sim.agent->ActiveTask()->Transition(g_model, g_data);

        // 运行中
        if (sim.run) {
          // 记录迭代开始的CPU时间
          const auto iteration_start = mj::Simulate::Clock::now();

          // 上次同步以来的CPU和仿真时间
          const auto elapsed_cpu = iteration_start - sync_cpu_time;
          const double elapsed_sim = g_data->time - sync_sim_time;

          // 注入噪声（Ornstein-Uhlenbeck过程）
          if (sim.ctrl_noise_std > 0.0) {
            // 将速率和尺度转换为离散时间
            const mjtNum rate = mju_exp(-g_model->opt.timestep / sim.ctrl_noise_rate);
            const mjtNum scale = sim.ctrl_noise_std * mju_sqrt(1 - rate * rate);

            for (int i = 0; i < g_model->nu; ++i) {
              // 更新噪声
              g_ctrl_noise[i] = rate * g_ctrl_noise[i] + 
                               scale * mju_standardNormal(nullptr);
            }
          }

          // 请求的降速因子
          const double slowdown = 100.0 / sim.percentRealTime[sim.real_time_index];

          // 不同步条件：目标仿真时间的距离大于最大不同步阈值
          const double cpu_time_seconds = Seconds(elapsed_cpu).count();
          const double misalignment = mju_abs(cpu_time_seconds / slowdown - elapsed_sim);
          const bool is_misaligned = misalignment > kSyncMisalignmentThreshold;

          // 不同步（任何原因）：重置同步时间，步进
          if (elapsed_sim < 0 || elapsed_cpu.count() < 0 ||
              sync_cpu_time.time_since_epoch().count() == 0 || 
              is_misaligned || sim.speed_changed) {
            // 重新同步
            sync_cpu_time = iteration_start;
            sync_sim_time = g_data->time;
            sim.speed_changed = false;

            // 运行单步，让下一次迭代处理定时
            SimulateStep(sim, g_model, g_data);
          } else {  // 同步中：步进直到超过CPU时间
            bool measured = false;
            const mjtNum previous_sim_time = g_data->time;
            const double refresh_time = kSimRefreshFraction / sim.refresh_rate;

            // 当仿真滞后于CPU且在refresh_time内时步进
            const auto start_cpu_time = iteration_start;
            
            while (Seconds((g_data->time - sync_sim_time) * slowdown) <
                       mj::Simulate::Clock::now() - sync_cpu_time &&
                   mj::Simulate::Clock::now() - start_cpu_time <
                       Seconds(refresh_time)) {
              // 在第一次步进前测量降速
              if (!measured && elapsed_sim > 0.0) {
                sim.measured_slowdown = cpu_time_seconds / elapsed_sim;
                measured = true;
              }

              // 执行模拟步进
              SimulateStep(sim, g_model, g_data);

              // 如果重置则跳出
              if (g_data->time < previous_sim_time) {
                break;
              }
            }
          }
        } else {  // 暂停中
          // 应用位姿扰动
          sim.ApplyPosePerturbations(1);  // 移动运动捕捉和动态体

          // 仿真暂停时仍然接受任务
          sim.agent->ExecuteAllRunBeforeStepJobs(g_model, g_data);

          // 运行mj_forward，更新渲染和关节滑块
          mj_forward(g_model, g_data);
          sim.speed_changed = true;
        }
      }
    }  // 释放sim.mtx

    // 状态更新
    if (sim.uiloadrequest.load() == 0) {
      // 如果没有活动估计器或估计器未启用，设置真实状态
      if (!sim.agent->ActiveEstimatorIndex() || !sim.agent->estimator_enabled) {
        sim.agent->state.Set(g_model, g_data);
      }
    }
  }
}
}  // namespace

// ------------------------------- 主函数 ----------------------------------------

namespace mjpc {

MjpcApp::MjpcApp(std::vector<std::shared_ptr<mjpc::Task>> tasks, int task_id) {
  // MJPC标题
  std::cout << "MuJoCo MPC (MJPC)\n";
  
  // MuJoCo版本信息
  std::cout << " MuJoCo version " << mj_versionString() << "\n";
  
  if (mjVERSION_HEADER != mj_version()) {
    mju_error("Headers and library have Different versions");
  }

  // 硬件线程信息
  std::cout << " Hardware threads:  " << mjpc::NumAvailableHardwareThreads() << "\n";

  if (g_sim != nullptr) {
    mju_error("Multiple instances of MjpcApp created.");
    return;
  }
  
  g_sim = std::make_unique<mj::Simulate>(
      std::make_unique<mujoco::GlfwAdapter>(),
      std::make_shared<Agent>());

  g_sim->agent->SetTaskList(std::move(tasks));
  g_sim->agent->gui_task_id = task_id;

  g_sim->filename = g_sim->agent->GetTaskXmlPath(g_sim->agent->gui_task_id);
  g_model = LoadModel(g_sim->agent.get(), *g_sim);
  
  if (g_model) {
    g_data = mj_makeData(g_model);
  }
  
  // 设置home关键帧
  if (g_model && g_data) {
    int home_id = mj_name2id(g_model, mjOBJ_KEY, "home");
    if (home_id >= 0) {
      mj_resetDataKeyframe(g_model, g_data, home_id);
    }
  }

  g_sim->mnew = g_model;
  g_sim->dnew = g_data;

  // 控制噪声分配
  free(g_ctrl_noise);
  if (g_model) {
    g_ctrl_noise = static_cast<mjtNum*>(malloc(sizeof(mjtNum) * g_model->nu));
    mju_zero(g_ctrl_noise, g_model->nu);
  }

  // 代理初始化
  if (g_model) {
    g_sim->agent->estimator_enabled = absl::GetFlag(FLAGS_estimator_enabled);
    g_sim->agent->Initialize(g_model);
    g_sim->agent->Allocate();
    g_sim->agent->Reset();
    g_sim->agent->PlotInitialize();
    g_sim->agent->plan_enabled = absl::GetFlag(FLAGS_planner_enabled);
  }

  // 获取最接近输入仿真百分比的索引
  float desired_percent = absl::GetFlag(FLAGS_sim_percent_realtime);
  const auto& percentages = g_sim->percentRealTime;
  auto closest = std::min_element(
      std::begin(percentages), std::end(percentages),
      [&](float a, float b) {
        return std::abs(a - desired_percent) < std::abs(b - desired_percent);
      });
  
  g_sim->real_time_index = std::distance(std::begin(percentages), closest);

  g_sim->delete_old_m_d = true;
  g_sim->loadrequest = 2;

  g_sim->ui0_enable = absl::GetFlag(FLAGS_show_left_ui);
  g_sim->info = absl::GetFlag(FLAGS_show_info);
}

MjpcApp::~MjpcApp() {
  g_sim.reset();
}

// 运行事件循环
void MjpcApp::Start() {
  // 线程信息输出
  std::cout << "  Physics        :  " << 1 << "\n";
  std::cout << "  Render         :  " << 1 << "\n";
  std::cout << "  Planner        :  " << 1 << "\n";
  std::cout << "    Planning     :  " << g_sim->agent->planner_threads() << "\n";
  std::cout << "  Estimator      :  " << g_sim->agent->estimator_threads() << "\n";
  std::cout << "    Estimation   :  " << g_sim->agent->estimator_enabled << "\n";

  // 设置控制回调
  mjcb_control = controller;

  // 设置传感器回调
  mjcb_sensor = sensor;

  // 一次性准备：初始化渲染循环
  g_sim->InitializeRenderLoop();

  // 启动物理线程
  mjpc::ThreadPool physics_pool(1);
  physics_pool.Schedule([]() { PhysicsLoop(*g_sim); });

  // 启动估计器线程
  mjpc::ThreadPool estimator_pool(1);
  if (g_sim->agent->estimator_enabled) {
    estimator_pool.Schedule([]() { EstimatorLoop(*g_sim); });
  }

  {
    // 启动规划线程
    mjpc::ThreadPool plan_pool(1);
    plan_pool.Schedule(
        []() { g_sim->agent->Plan(g_sim->exitrequest, g_sim->uiloadrequest); });

    // 现在规划已经分叉，主线程可以渲染

    // 启动仿真UI循环（阻塞调用）
    g_sim->RenderLoop();
  }
}

mj::Simulate* MjpcApp::Sim() {
  return g_sim.get();
}

void StartApp(std::vector<std::shared_ptr<mjpc::Task>> tasks, int task_id) {
  MjpcApp app(std::move(tasks), task_id);
  app.Start();
}

}  // namespace mjpc
