# 最低版本要求:
1. Python : 3.11.3
2. pygame : 2.6.0
3. numpy : 2.2.6

# 生物邏輯
## 所有生物
1. 移動
2. 與障礙物碰撞
3. 穿過邊緣

## Bird
### 與 Bird 互動
1. 分離，避免太擁擠
2. 對齊，嘗試跟上其他同類並往同方向移動
3. 聚集，與同類靠近
### 與 Predator 互動
1. 逃離，遠離 Predator

## Predator
### 與 Predator 互動
1. 分離，避免太擁擠
### 與 Bird 互動
1. 追擊，往視野範圍內最多 Bird 的地方前進

# 參考資料
1. **Gemini 2.5 Flash** 以下簡稱 AI
2. https://boids.dan.onl/ 以下簡稱 online boids

## class Animal
### 屬性
自行設計
### 方法
#### draw (繪圖)
自行設計，AI實作
#### apply_bounce (碰撞處理)  
AI設計，AI實作
#### basis_update (更新狀態)
自行設計，自行實作

## class Bird (繼承自 Animal)
### 屬性
自行設計
### 方法
#### apply_force (boids 三種規則)
AI設計，AI實作，自行整合
#### flee_predator (逃離掠食者)
自行設計，自行實作，參考自AI設計的三種規則
#### update (更新狀態)
自行設計，自行實作
## class Predator (繼承自 Animal)
### 屬性
自行設計
### 方法
#### apply_track (追蹤 bird)
自行設計，自行實作，參考自AI設計的三種規則
#### apply_separation (與其他 Predator 分離)
自行設計，自行實作，參考自AI設計的三種規則
#### update (更新狀態)
自行設計，自行實作

## class Obstacle
### 屬性
AI設計
### 方法
#### draw (繪圖)
AI設計，AI實作
#### check_collision (圓形碰撞處理)
AI設計，AI實作
#### generate_random_polygon (生成多邊形外型)
AI設計，AI實作

## 其他
### 細節
1. bird 隨速度變色 : 方便觀察，自行設計和實作，參考自 [online boids](https://boids.dan.onl/)
2. 阻力 : 自行設計和實作
3. 碰撞時能量遞減 : 自行設計和實作
### 優化
1. Movement_Accuracy : 提升效能，自行設計和實作，參考自 [online boids](https://boids.dan.onl/)

### 其餘皆自行設計與實作
