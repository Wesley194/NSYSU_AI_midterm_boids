# 最低版本要求:
1. Python : 3.11.3
2. pygame : 2.6.0
3. numpy : 2.2.6
4. ttkbootstrap : 1.16.0

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
2. 死亡，被 Predator 追上，產生死亡動畫，在邊界復活
### 與滑鼠互動
1. 按住滑鼠右鍵聚集附近的 Bird
2. 按住滑鼠左鍵排斥附近的 Bird

## Predator
### 與 Predator 互動
1. 分離，避免太擁擠
#### 與 Bird 互動
1. 追擊，往視野範圍內最多 Bird 的地方前進

# 參考資料
1. **Gemini 2.5 Flash** 以下簡稱 AI
2. https://boids.dan.onl/ 以下簡稱 online boids

# 貢獻表
## class Animal
### 屬性
自行設計
### 方法
#### draw (繪圖)
  - AI實作
  - 三角形標示方向參考 [online boids](https://boids.dan.onl/)
#### apply_bounce (碰撞處理)  
  - AI設計實作
  - 嘗試過多邊形碰撞，但後來改用效率較佳的圓形碰撞
#### basis_update (更新狀態)
自行設計實作

## class Bird (繼承自 Animal)
### 屬性
自行設計實作
### 方法
#### apply_force (boids 三種規則)
  - AI設計實作
  - 自行整合優化
#### flee_predator (逃離掠食者)
自行設計實作
#### mouse_activity (Bird 靠近或遠離滑鼠)
自行設計實作
#### update (更新狀態)
自行設計實作

## class Predator (繼承自 Animal)
全部都是人類設計和實作

## class Particle
全部都是人類設計和實作

## class Obstacle
AI設計實作

## ttk 調參數視窗
  - AI實作
  - 自行整合優化
## 其他
### 值得一提的功能
以下皆是人類設計實作
1. bird 隨速度變色 : 方便觀察，參考自 [online boids](https://boids.dan.onl/)
2. 阻力
3. 碰撞時能量遞減
4. bird 死亡互動
5. UI 參數調控
6. 生物動態移除與生成
### 優化
以下皆是人類修改實作
1. Movement_Accuracy : 提升效能，參考自 [online boids](https://boids.dan.onl/)
2. 計算向量長度時盡量使用 length_squared() 而非 length() ，減少開根號發生
3. 移除不必要的執行序鎖(AI寫的比較嚴緊，反而導致效率低落)

### 其餘程式皆自行設計實作
