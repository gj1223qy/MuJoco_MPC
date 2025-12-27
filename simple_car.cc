// Copyright 2022 DeepMind Technologies Limited
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

#include "mjpc/tasks/simple_car/simple_car.h"

#include <cmath>
#include <cstdio>
#include <string>
#include <sstream>
#include <iomanip>

#include <absl/random/random.h>
#include <mujoco/mujoco.h>
#include "mjpc/task.h"
#include "mjpc/utilities.h"
#include <GL/gl.h>
#include <GL/glu.h>
#include <GL/freeglut.h>


namespace mjpc {

std::string SimpleCar::XmlPath() const {
  return GetModelPath("simple_car/task.xml");
}

std::string SimpleCar::Name() const { return "SimpleCar"; }

float speed = 0.0f;  // 当前车速（km/h）


// 绘制圆形边框
void SimpleCar::drawCircle(float radius, int segments) {
    glBegin(GL_LINE_LOOP);
    for (int i = 0; i < segments; i++) {
        float angle = 2 * M_PI * i / segments;
        glVertex2f(radius * cos(angle), radius * sin(angle));
    }
    glEnd();
}

void SimpleCar::drawTicks(float radius, int tickCount) {
    float angleStep = 180.0f / tickCount;  // 因为是半圆，刻度分布在180度内
    for (int i = 0; i < tickCount; i++) {
        float angle = i * angleStep * M_PI / 180.0f;
        float tickLength = 0.02f;
        float x1 = radius * cos(angle);
        float y1 = radius * sin(angle);
        float x2 = (radius - tickLength) * cos(angle);
        float y2 = (radius - tickLength) * sin(angle);
        
        glBegin(GL_LINES);
        glVertex2f(x1, y1);
        glVertex2f(x2, y2);
        glEnd();
    }
}

// 绘制速度指针
void SimpleCar::drawPointer(float angle) {
    glBegin(GL_LINES);
    glVertex2f(0.0f, 0.0f);  // 指针的根部
    glVertex2f(0.8f * cos(angle), 0.8f * sin(angle));  // 指针的尖端
    glEnd();
}

// 绘制数字
void SimpleCar::drawNumber(float radius, int number, float angle) {
    char buffer[10];
    snprintf(buffer, sizeof(buffer), "%d", number);
    glRasterPos2f(radius * cos(angle), radius * sin(angle));
    for (int i = 0; buffer[i] != '\0'; i++) {
        glutBitmapCharacter(GLUT_BITMAP_HELVETICA_18, buffer[i]);
    }
}

// 绘制仪表盘
void SimpleCar::drawDashboard(float* dashboard_pos, float speed_ratio) {
   glClear(GL_COLOR_BUFFER_BIT);

    // 将仪表盘移动到正确的位置
    glPushMatrix();
    glTranslatef(dashboard_pos[0], dashboard_pos[1], dashboard_pos[2]);

    // 绘制外圈（仪表盘圆形）
    glColor3f(0.1f, 0.1f, 0.1f);  // 深灰色边框
    drawCircle(0.6f, 100);

    // 绘制刻度线（0, 2, 4, 6, 8, 10 共6个刻度）
    glColor3f(1.0f, 1.0f, 1.0f);  // 白色刻度线
    drawTicks(0.5f, 10);  // 总共绘制10个刻度

    // 绘制速度指针
    float pointerAngle = (90.0f - (180.0f * speed_ratio)) * M_PI / 180.0f;  // 根据车速计算角度
    glColor3f(1.0f, 0.0f, 0.0f);  // 红色指针
    drawPointer(pointerAngle);

    // 绘制刻度数字（0, 1, 2, ..., 10）
    for (int i = 0; i <= 10; i++) {
        float angle = (90.0f - 18.0f * i) * M_PI / 180.0f;
        drawNumber(0.45f, i, angle);
    }

    // 绘制"km/h"单位
    glColor3f(0.9f, 0.9f, 0.9f);
    glRasterPos2f(0.0f, -0.7f);
    const char* unit = "km/h";
    for (int i = 0; unit[i] != '\0'; i++) {
        glutBitmapCharacter(GLUT_BITMAP_HELVETICA_18, unit[i]);
    }

    // 绘制当前速度值
    glColor3f(0.9f, 0.9f, 0.9f);
    char speedStr[50];
    snprintf(speedStr, sizeof(speedStr), "%.1f", speed);
    glRasterPos2f(-0.1f, 0.0f);
    for (int i = 0; speedStr[i] != '\0'; i++) {
        glutBitmapCharacter(GLUT_BITMAP_HELVETICA_18, speedStr[i]);
    }

    // 恢复状态
    glPopMatrix();
    glutSwapBuffers();
}


// ------- Residuals for simple_car task ------
//     Position: Car should reach goal position (x, y)
//     Control:  Controls should be small
// ------------------------------------------
void SimpleCar::ResidualFn::Residual(const mjModel* model, const mjData* data,
                                     double* residual) const {
  // ---------- Position (x, y) ----------
  // Goal position from mocap body
  residual[0] = data->qpos[0] - data->mocap_pos[0];  // x position
  residual[1] = data->qpos[1] - data->mocap_pos[1];  // y position

  // ---------- Control ----------
  residual[2] = data->ctrl[0];  // forward control
  residual[3] = data->ctrl[1];  // turn control
}

// -------- Transition for simple_car task --------
//   If car is within tolerance of goal ->
//   move goal randomly.
// ------------------------------------------------
void SimpleCar::TransitionLocked(mjModel* model, mjData* data) {
  // Car position (x, y)
  double car_pos[2] = {data->qpos[0], data->qpos[1]};
  
  // Goal position from mocap
  double goal_pos[2] = {data->mocap_pos[0], data->mocap_pos[1]};
  
  // Distance to goal
  double car_to_goal[2];
  mju_sub(car_to_goal, goal_pos, car_pos, 2);
  
  // If within tolerance, move goal to random position
  if (mju_norm(car_to_goal, 2) < 0.2) {
    absl::BitGen gen_;
    data->mocap_pos[0] = absl::Uniform<double>(gen_, -2.0, 2.0);
    data->mocap_pos[1] = absl::Uniform<double>(gen_, -2.0, 2.0);
    data->mocap_pos[2] = 0.01;  // keep z at ground level
  }
}

// draw task-related geometry in the scene
// 改进后的立式仪表盘，放置在汽车正上方

void SimpleCar::ModifyScene(const mjModel* model, const mjData* data,
                            mjvScene* scene) const {
    // 改进的控制台输出格式
    static double fuel_capacity = 100.0;   // 满油 = 100 单位
    static double fuel_used = 0.0;    // 累计油耗（任意单位）

    // 1. 位置
    double pos_x = data->qpos[0];
    double pos_y = data->qpos[1];

    // 2. 速度 - 直接计算总速度，不单独存储x,y分量
    double speed_ms = mju_norm(data->qvel, 2);  // 总速度大小

    // 3. 加速度 - 直接计算总加速度
    double acc_mag = mju_norm(data->qacc, 2);  // 加速度大小

    // 4. 车体速度（用于转速）
    double* car_velocity = SensorByName(model, data, "car_velocity");
    double speed_ms_sensor = car_velocity ? mju_norm3(car_velocity) : 0.0;

    // 5. 转速条（30 个 #）
    const int BAR_LEN = 30;
    const double max_speed_ref = 5.0;   // 参考最大速度
    double rpm_ratio = speed_ms_sensor / max_speed_ref;
    if (rpm_ratio > 1.0) rpm_ratio = 1.0;
    if (rpm_ratio < 0.0) rpm_ratio = 0.0;

    int filled = static_cast<int>(rpm_ratio * BAR_LEN);

    char rpm_bar[BAR_LEN + 1];
    for (int i = 0; i < BAR_LEN; i++) {
        rpm_bar[i] = (i < filled) ? '#' : ' ';
    }
    rpm_bar[BAR_LEN] = '\0';

    double dt = model->opt.timestep;

    // 油门控制（前进控制）
    double throttle = data->ctrl[0];

    // 油耗系数
    const double fuel_coeff = 0.2;

    // 累计油耗
    fuel_used += fuel_coeff * std::abs(throttle) * dt;

    // 不允许超过油箱容量
    if (fuel_used > fuel_capacity) {
        fuel_used = fuel_capacity;
    }
    double fuel_left = fuel_capacity - fuel_used;
    double fuel_percent = (fuel_left / fuel_capacity) * 100.0;

    // 防止数值异常
    if (fuel_percent < 0.0) fuel_percent = 0.0;
    if (fuel_percent > 100.0) fuel_percent = 100.0;

    // 改进的格式化输出
    printf("\r");
    printf("位置: (%.2f, %.2f) | ", pos_x, pos_y);
    printf("速度: %.2f m/s (%.1f km/h) | ", speed_ms, speed_ms * 3.6);
    printf("加速度: %.2f m/s² | ", acc_mag);
    printf("油耗: %3.0f%% | ", fuel_percent);
    printf("转速: [%s]", rpm_bar);
    fflush(stdout);

    // 获取汽车车身ID
    int car_body_id = mj_name2id(model, mjOBJ_BODY, "car");
    if (car_body_id < 0) {
        printf("\n[警告] 未找到汽车车身 'car'\n");
        return;  // 汽车车身未找到
    }
  
    // 计算速度
    double* car_pos = data->xpos + 3 * car_body_id;
    double speed_kmh = speed_ms * 3.6;  // 将m/s转换为km/h
  
    // 仪表盘位置（汽车正上方，提高高度到1.2米，更可见）
    float dashboard_pos[3] = {
        static_cast<float>(car_pos[0]),
        static_cast<float>(car_pos[1] + 0.2f),  // 汽车前方0.2米
        static_cast<float>(car_pos[2] + 1.2f)   // 地面上方1.2米（提高高度）
    };

    const float gauge_scale = 2.5f;  // 仪表盘整体放大
    const float max_speed_kmh = 10.0f;  // 最大速度参考值

    // 速度百分比（0-1）
    float speed_ratio = static_cast<float>(speed_kmh) / max_speed_kmh;
    if (speed_ratio > 1.0f) speed_ratio = 1.0f;
    if (speed_ratio < 0.0f) speed_ratio = 0.0f;
    
    printf("\n[调试] 车速: %.2f km/h, 速度比例: %.2f\n", speed_kmh, speed_ratio);
  
    // 仪表盘旋转矩阵（绕X轴旋转90度，再顺时针旋转90度）
    double angle_x = 90.0 * 3.14159 / 180.0;  // 绕X轴旋转90度（立起来）
    double cos_x = cos(angle_x);
    double sin_x = sin(angle_x);
    double mat_x[9] = {
        1, 0,      0,
        0, cos_x, -sin_x,
        0, sin_x,  cos_x
    };
  
    double angle_z = -90.0 * 3.14159 / 180.0;  // 绕Z轴旋转-90度（顺时针）
    double cos_z = cos(angle_z);
    double sin_z = sin(angle_z);
    double mat_z[9] = {
        cos_z, -sin_z, 0,
        sin_z,  cos_z, 0,
        0,      0,     1
    };
  
    // 组合旋转矩阵：先绕X轴旋转90°，再绕Z轴顺时针旋转90°
    double dashboard_rot_mat[9];
    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 3; j++) {
            dashboard_rot_mat[i*3 + j] = 0;
            for (int k = 0; k < 3; k++) {
                dashboard_rot_mat[i*3 + j] += mat_z[i*3 + k] * mat_x[k*3 + j];
            }
        }
    }

    // 1. 仪表盘背景（半透明深色圆盘）
    if (scene->ngeom < scene->maxgeom) {
        mjvGeom* geom = scene->geoms + scene->ngeom;
        
        geom->type = mjGEOM_CYLINDER;
        geom->size[0] = geom->size[1] = 0.15f * gauge_scale;
        geom->size[2] = 0.008f * gauge_scale;
        
        geom->pos[0] = dashboard_pos[0];
        geom->pos[1] = dashboard_pos[1];
        geom->pos[2] = dashboard_pos[2];
        
        // 应用仪表盘旋转矩阵
        for (int j = 0; j < 9; j++) {
            geom->mat[j] = static_cast<float>(dashboard_rot_mat[j]);
        }
        
        // 深色半透明背景
        geom->rgba[0] = 0.08f;
        geom->rgba[1] = 0.08f;
        geom->rgba[2] = 0.12f;
        geom->rgba[3] = 0.85f;
        scene->ngeom++;
        printf("[调试] 添加仪表盘背景\n");
    }

    // 2. 仪表盘外边框（金属质感）
    if (scene->ngeom < scene->maxgeom) {
        mjvGeom* geom = scene->geoms + scene->ngeom;
        
        geom->type = mjGEOM_CYLINDER;
        geom->size[0] = geom->size[1] = 0.152f * gauge_scale;
        geom->size[2] = 0.003f * gauge_scale;
        
        geom->pos[0] = dashboard_pos[0];
        geom->pos[1] = dashboard_pos[1];
        geom->pos[2] = dashboard_pos[2] + 0.002f;
        
        for (int j = 0; j < 9; j++) {
            geom->mat[j] = static_cast<float>(dashboard_rot_mat[j]);
        }
        
        // 金属银色边框
        geom->rgba[0] = 0.8f;
        geom->rgba[1] = 0.85f;
        geom->rgba[2] = 0.9f;
        geom->rgba[3] = 0.9f;
        scene->ngeom++;
    }

    // 3. 彩色速度弧段（从蓝到红）
    const int ARC_SEG = 40;
    for (int s = 0; s < ARC_SEG; s++) {
        if (scene->ngeom >= scene->maxgeom) break;

        float t0 = (ARC_SEG <= 1) ? 0.0f : (float)s / (float)(ARC_SEG - 1);
        float angle_deg = 180.0f - 180.0f * t0;
        float rad = angle_deg * 3.14159f / 180.0f;

        float arc_r = 0.135f * gauge_scale;
        float arc_y = dashboard_pos[1] - arc_r * (float)cos(rad);
        float arc_z = dashboard_pos[2] + arc_r * (float)sin(rad);

        // 旋转矩阵
        double rot_deg = angle_deg - 90.0;
        double rr = rot_deg * 3.14159 / 180.0;
        double c = cos(rr), si = sin(rr);
        double zrot[9] = { c, -si, 0, si, c, 0, 0, 0, 1 };

        double matd[9];
        for (int r = 0; r < 3; r++) {
            for (int cc = 0; cc < 3; cc++) {
                matd[r*3 + cc] = 0.0;
                for (int k = 0; k < 3; k++) {
                    matd[r*3 + cc] += dashboard_rot_mat[r*3 + k] * zrot[k*3 + cc];
                }
            }
        }

        // 改进的颜色渐变：蓝色 -> 青色 -> 绿色 -> 黄色 -> 红色
        float rcol, gcol, bcol;
        if (t0 < 0.2f) {      // 蓝色段
            float tt = t0 / 0.2f;
            rcol = 0.1f;
            gcol = 0.3f + 0.5f * tt;
            bcol = 1.0f;
        } else if (t0 < 0.4f) { // 青色段
            float tt = (t0 - 0.2f) / 0.2f;
            rcol = 0.1f;
            gcol = 0.8f + 0.1f * tt;
            bcol = 1.0f - 0.3f * tt;
        } else if (t0 < 0.6f) { // 绿色段
            float tt = (t0 - 0.4f) / 0.2f;
            rcol = 0.1f + 0.3f * tt;
            gcol = 0.9f;
            bcol = 0.7f - 0.5f * tt;
        } else if (t0 < 0.8f) { // 黄色段
            float tt = (t0 - 0.6f) / 0.2f;
            rcol = 0.4f + 0.4f * tt;
            gcol = 0.9f;
            bcol = 0.2f - 0.2f * tt;
        } else {              // 红色段
            float tt = (t0 - 0.8f) / 0.2f;
            rcol = 0.8f + 0.2f * tt;
            gcol = 0.9f - 0.6f * tt;
            bcol = 0.0f;
        }

        mjvGeom* g = scene->geoms + scene->ngeom;
        mjtNum size[3] = {
            (mjtNum)(0.0025f * gauge_scale),
            (mjtNum)(0.0080f * gauge_scale),
            (mjtNum)(0.0025f * gauge_scale)
        };

        mjtNum pos[3] = {
            (mjtNum)dashboard_pos[0],
            (mjtNum)arc_y,
            (mjtNum)arc_z
        };

        mjtNum mat9[9];
        for (int j = 0; j < 9; j++) mat9[j] = (mjtNum)matd[j];

        float rgba[4] = { rcol, gcol, bcol, 0.9f };
        mjv_initGeom(g, mjGEOM_BOX, size, pos, mat9, rgba);
        scene->ngeom++;
    }

    // 4. 刻度线（0~10）
    const int kMaxTick = 10;
    const int kTickCount = kMaxTick + 1;
    
    for (int i = 0; i < kTickCount; i++) {
        if (scene->ngeom >= scene->maxgeom) break;

        int tick_value = i;
        float tick_angle_deg = 180.0f - (180.0f * tick_value / kMaxTick);
        float rad_tick_angle = tick_angle_deg * 3.14159f / 180.0f;

        float full_len = ((tick_value % 5 == 0) ? 0.025f : 0.015f) * gauge_scale;
        float half_len = full_len * 0.5f;
        float tick_radius_outer = 0.130f * gauge_scale;
        float tick_radius_center = tick_radius_outer - half_len;

        mjvGeom* geom = scene->geoms + scene->ngeom;
        geom->type = mjGEOM_BOX;
        geom->size[0] = 0.0012f * gauge_scale;
        geom->size[1] = half_len;
        geom->size[2] = 0.0012f * gauge_scale;

        float tick_y = dashboard_pos[1] - tick_radius_center * cos(rad_tick_angle);
        float tick_z = dashboard_pos[2] + tick_radius_center * sin(rad_tick_angle);

        geom->pos[0] = dashboard_pos[0];
        geom->pos[1] = tick_y;
        geom->pos[2] = tick_z;

        // 刻度指向圆心
        double tick_rot_angle = tick_angle_deg - 90.0;
        double rad_tick_rot = tick_rot_angle * 3.14159 / 180.0;
        double cos_t = cos(rad_tick_rot);
        double sin_t = sin(rad_tick_rot);

        double tick_rot_mat[9] = { cos_t, -sin_t, 0, sin_t, cos_t, 0, 0, 0, 1 };
        double tick_mat[9];
        for (int r = 0; r < 3; r++) {
            for (int c = 0; c < 3; c++) {
                tick_mat[r*3 + c] = 0;
                for (int k = 0; k < 3; k++) {
                    tick_mat[r*3 + c] += dashboard_rot_mat[r*3 + k] * tick_rot_mat[k*3 + c];
                }
            }
        }

        for (int j = 0; j < 9; j++) {
            geom->mat[j] = static_cast<float>(tick_mat[j]);
        }

        // 刻度颜色：主刻度白色，次要刻度灰色
        if (tick_value % 5 == 0) {
            geom->rgba[0] = 1.0f;
            geom->rgba[1] = 1.0f;
            geom->rgba[2] = 1.0f;
        } else {
            geom->rgba[0] = 0.7f;
            geom->rgba[1] = 0.7f;
            geom->rgba[2] = 0.7f;
        }
        geom->rgba[3] = 0.9f;
        scene->ngeom++;

        // 数字标签（只显示0, 2, 4, 6, 8, 10）
        if (tick_value % 2 == 0 && scene->ngeom < scene->maxgeom) {
            mjvGeom* label_geom = scene->geoms + scene->ngeom;
            label_geom->type = mjGEOM_LABEL;
            label_geom->size[0] = label_geom->size[1] = label_geom->size[2] = 0.045f * gauge_scale;

            float label_radius = 0.17f * gauge_scale;
            label_geom->pos[0] = dashboard_pos[0];
            label_geom->pos[1] = dashboard_pos[1] - label_radius * cos(rad_tick_angle);
            label_geom->pos[2] = dashboard_pos[2] + label_radius * sin(rad_tick_angle);
            
            label_geom->rgba[0] = 0.95f;
            label_geom->rgba[1] = 0.95f;
            label_geom->rgba[2] = 0.95f;
            label_geom->rgba[3] = 1.0f;

            std::snprintf(label_geom->label, sizeof(label_geom->label), "%d", tick_value);
            scene->ngeom++;
        }
    }

    // 5. 速度指针（增强设计，更明显）
    if (scene->ngeom < scene->maxgeom) {
        mjvGeom* geom = scene->geoms + scene->ngeom;
        geom->type = mjGEOM_BOX;
        geom->size[0] = 0.005f * gauge_scale;  // 加粗指针
        geom->size[1] = 0.110f * gauge_scale;  // 加长指针
        geom->size[2] = 0.005f * gauge_scale;

        // 指针角度：0速度时指向180度（左），满速度时指向0度（右）
        float angle = 180.0f - 180.0f * speed_ratio;
        printf("[调试] 指针角度: %.1f 度\n", angle);
        float rad_angle = angle * 3.14159f / 180.0f;
        
        // 指针位置：稍微偏移，让指针根部在中心点
        float pointer_y = dashboard_pos[1] - 0.040f * gauge_scale * cos(rad_angle);
        float pointer_z = dashboard_pos[2] + 0.040f * gauge_scale * sin(rad_angle);

        geom->pos[0] = dashboard_pos[0];
        geom->pos[1] = pointer_y;
        geom->pos[2] = pointer_z;

        // 指针旋转矩阵
        double pointer_angle = angle - 90.0;
        double rad_pointer_angle = pointer_angle * 3.14159 / 180.0;
        double cos_p = cos(rad_pointer_angle);
        double sin_p = sin(rad_pointer_angle);
        double pointer_rot_mat[9] = { cos_p, -sin_p, 0, sin_p, cos_p, 0, 0, 0, 1 };

        double temp_mat[9];
        for (int i = 0; i < 3; i++) {
            for (int j = 0; j < 3; j++) {
                temp_mat[i*3 + j] = 0;
                for (int k = 0; k < 3; k++) {
                    temp_mat[i*3 + j] += dashboard_rot_mat[i*3 + k] * pointer_rot_mat[k*3 + j];
                }
            }
        }

        for (int i = 0; i < 9; i++) {
            geom->mat[i] = static_cast<float>(temp_mat[i]);
        }

        // 指针颜色：亮红色，非常明显
        geom->rgba[0] = 1.0f;  // 最亮的红色
        geom->rgba[1] = 0.0f;
        geom->rgba[2] = 0.0f;
        geom->rgba[3] = 1.0f;  // 完全不透明
        scene->ngeom++;
        printf("[调试] 添加速度指针\n");
    }

    // 6. 中心固定点（改进设计）
    if (scene->ngeom < scene->maxgeom) {
        mjvGeom* geom = scene->geoms + scene->ngeom;
        geom->type = mjGEOM_SPHERE;
        geom->size[0] = geom->size[1] = geom->size[2] = 0.010f * gauge_scale;  // 加大

        geom->pos[0] = dashboard_pos[0];
        geom->pos[1] = dashboard_pos[1];
        geom->pos[2] = dashboard_pos[2];
        
        for (int j = 0; j < 9; j++) {
            geom->mat[j] = static_cast<float>(dashboard_rot_mat[j]);
        }
        
        // 亮银色中心点
        geom->rgba[0] = 1.0f;
        geom->rgba[1] = 1.0f;
        geom->rgba[2] = 1.0f;
        geom->rgba[3] = 1.0f;
        scene->ngeom++;
    }

    // 7. 数字速度显示（改进位置和样式）
    if (scene->ngeom < scene->maxgeom) {
        mjvGeom* geom = scene->geoms + scene->ngeom;
        geom->type = mjGEOM_LABEL;
        geom->size[0] = geom->size[1] = geom->size[2] = 0.08f * gauge_scale;  // 加大字体
        geom->pos[0] = dashboard_pos[0];
        geom->pos[1] = dashboard_pos[1];
        geom->pos[2] = dashboard_pos[2] + 0.08f;  // 向上移动
        
        // 根据速度改变颜色
        if (speed_ratio < 0.3f) {
            geom->rgba[0] = 0.4f;  // 低速蓝色
            geom->rgba[1] = 0.6f;
            geom->rgba[2] = 1.0f;
        } else if (speed_ratio < 0.7f) {
            geom->rgba[0] = 0.9f;  // 中速绿色
            geom->rgba[1] = 0.9f;
            geom->rgba[2] = 0.4f;
        } else {
            geom->rgba[0] = 1.0f;  // 高速红色
            geom->rgba[1] = 0.5f;
            geom->rgba[2] = 0.5f;
        }
        geom->rgba[3] = 1.0f;
        
        std::stringstream ss;
        ss << std::fixed << std::setprecision(1) << speed_kmh;
        std::strncpy(geom->label, ss.str().c_str(), sizeof(geom->label) - 1);
        geom->label[sizeof(geom->label) - 1] = '\0';
        scene->ngeom++;
        printf("[调试] 速度显示: %.1f km/h\n", speed_kmh);
    }

    // 8. 添加"km/h"单位标签（改进位置）
    if (scene->ngeom < scene->maxgeom) {
        mjvGeom* geom = scene->geoms + scene->ngeom;
        geom->type = mjGEOM_LABEL;
        geom->size[0] = geom->size[1] = geom->size[2] = 0.05f * gauge_scale;
        geom->pos[0] = dashboard_pos[0];
        geom->pos[1] = dashboard_pos[1];
        geom->pos[2] = dashboard_pos[2] - 0.10f;  // 向下移动
        
        geom->rgba[0] = 0.9f;
        geom->rgba[1] = 0.9f;
        geom->rgba[2] = 0.9f;
        geom->rgba[3] = 1.0f;
        
        std::strncpy(geom->label, "km/h", sizeof(geom->label) - 1);
        geom->label[sizeof(geom->label) - 1] = '\0';
        scene->ngeom++;
    }

    // 9. 添加速度警告指示（当速度过高时显示）
    if (speed_ratio > 0.8f && scene->ngeom < scene->maxgeom) {
        mjvGeom* geom = scene->geoms + scene->ngeom;
        geom->type = mjGEOM_SPHERE;
        geom->size[0] = geom->size[1] = geom->size[2] = 0.015f * gauge_scale;  // 加大
        
        geom->pos[0] = dashboard_pos[0];
        geom->pos[1] = dashboard_pos[1] + 0.15f;
        geom->pos[2] = dashboard_pos[2];
        
        for (int j = 0; j < 9; j++) {
            geom->mat[j] = static_cast<float>(dashboard_rot_mat[j]);
        }
        
        // 闪烁效果：根据时间变化透明度
        static float blink = 0.0f;
        blink += 0.1f;
        float alpha = 0.5f + 0.5f * sin(blink);
        
        geom->rgba[0] = 1.0f;
        geom->rgba[1] = 0.0f;
        geom->rgba[2] = 0.0f;
        geom->rgba[3] = alpha;
        scene->ngeom++;
        printf("[调试] 高速警告显示\n");
    }
    
    printf("[调试] 仪表盘总数: %d\n", scene->ngeom);
}

}  // namespace mjpc
