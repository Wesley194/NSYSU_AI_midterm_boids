import pygame
import numpy.random as random
import numpy as np
import ttkbootstrap as ttk
import threading


#setting
SCREEN_WIDTH = 1280
SCREEN_HEIGHT = 720
SCREEN_CNETER=(SCREEN_WIDTH/2,SCREEN_HEIGHT/2)
BACKGROUND_COLOR = (25, 25, 25)

stop_event = threading.Event()

# pygame
def run_pygame(shared_state, state_lock, stop_event):

    #初始化
    pygame.init()
    Screen = pygame.display.set_mode((SCREEN_WIDTH,SCREEN_HEIGHT))
    pygame.display.set_caption('boids V3.0.0')
    Timer = pygame.time.Clock()

    #載入圖片、字體
    # Font = os.path.join("TaipeiSansTCBeta-Regular.ttf")

    #全域變數
    Damping = 2e-3 #阻力

    Bird_Number = 300 #bird 數量
    Bird_Size = 8 #bird 大小
    Bird_MAX_Speed = 200 #bird 最大速度
    Bird_MIN_Speed = 20 #bird 最小速度
    Bird_Color_Slow = pygame.math.Vector3(75,76,255) #bird 最慢速顏色
    Bird_Color_Fast = pygame.math.Vector3(63,255,50) #bird 最快速顏色
    Bird_Color_ChangeRate = Bird_Color_Fast-Bird_Color_Slow
    Bird_Perception_Radius = 30 #bird 觀察範圍
    Bird_Separation_Weight = 1 #bird 分離力最大值
    Bird_Alignment_Weight = 1 #bird 對齊力最大值
    Bird_Cohesion_Weight = 1 #bird 聚集力最大值
    Bird_Flee_Weight = 4 #bird 逃跑力最大值
    Bird_Alert_Radius = 50 #bird 警戒範圍
    Bird_Mouse_Activity = {"pos":(0,0),"click":0}

    Predator_Number = 3 #Predator 數量
    Predator_Size = 10 #Predator 大小
    Predator_MIN_Speed = 60 #Predator 最小速度
    Predator_MAX_Speed = 170 #Predator 最大速度
    Predator_Perception_Radius = 60 #Predator 觀察範圍
    Predator_Track_Weight = 2 #Predator 追蹤力
    Predator_Separation_Weight = 1 #Predator 分離力
    Predator_Eat_Radius = 8 #Predator 捕食範圍

    Particle_Size = 10 #粒子大小
    Particle_Num = 3 #每次動物死亡，出現的粒子數
    Particle_Lifetime = 25 #粒子的生命週期 (tick)
    Particle_Radious = 1 #粒子半徑
    Particle_MIN_Speed = 15 #粒子的最小速度
    Particle_MAX_Speed = 25 #粒子的最大速度
    Particle_offset_angle = 120 #粒子init後和輸入的angle最大差值

    Obstacle_Number = 4 # Obstacle 數量
    Obstacle_Size = 80 # Obstacle 大小
    Bounce_Damping = 0.8 # bird 碰撞時能量遞減

    # 計算精度，若有 n 隻 bird ，則每隻 bird 需要與 n-1 隻 bird 互動，
    # 為提升效能我這裡只讓 bird 與隨機 (n-1)*Movement_Accuracy 隻 bird 互動
    # 另一種想法是讓每隻 bird 有 Movement_Accuracy 的機率"不合群"，違背自然法則，模擬自然的隨機性
    Movement_Accuracy = 0.5

    DT=0 #每楨之間時間間隔，確保不同楨率下動畫表現一致

    #物件定義
    class Animal:
        def __init__(self,pos,size,speed,color=(255, 255, 255)):
            self.position = pygame.math.Vector2(pos['x'], pos['y']) #初始位置
            self.direction=pygame.math.Vector2(0,0)
            self.direction.from_polar((1, random.uniform(1,360)))#初始方向
            self.speed = speed #初始速率
            self.velocity = self.direction*self.speed #初始速度
            self.color = color #顏色
            self.size = size #大小
            self.situation = "alive" #個體目前的狀態，有 1.alive 2.dying(做死亡動畫) 3.dead(即將重生)

        def draw(self, screen):
            right_vector = pygame.math.Vector2(-self.direction.y, self.direction.x) #垂直於 self.direction 的向量
            #三個頂點
            head = self.position + self.direction * self.size * 1
            left = self.position - self.direction * self.size + right_vector * self.size * 0.5
            right = self.position - self.direction * self.size - right_vector * self.size * 0.5
            #繪製三角形
            pygame.draw.polygon(screen, self.color, 
                                [(int(head.x), int(head.y)), 
                                 (int(left.x), int(left.y)), 
                                 (int(right.x), int(right.y))])

        def apply_bounce(self, obstacles):
            """
            迭代所有障礙物，並呼叫其 check_collision 方法來處理碰撞和反彈。
            """
            for obstacle in obstacles:
                #將自己（boid 實例）傳遞給障礙物物件
                if obstacle.check_collision(self):
                    #處理單一碰撞後立即退出，避免 boid 被多個障礙物邊緣重複處理
                    return
        
        def basis_update(self, MAX_SPEED, MIN_SPPED, obstacles):
            #碰撞反彈
            self.apply_bounce(obstacles)

            #更改速度
            self.speed *= (1 - Damping)
            self.speed = min(max(self.speed, MIN_SPPED), MAX_SPEED)
            self.velocity = self.direction * self.speed * DT

            #更新位置
            self.position += self.velocity

            #邊界處理
            if self.position.x > SCREEN_WIDTH:
                self.position.x = 0
            elif self.position.x < 0:
                self.position.x = SCREEN_WIDTH

            if self.position.y > SCREEN_HEIGHT:
                self.position.y = 0
            elif self.position.y < 0:
                self.position.y = SCREEN_HEIGHT

    class Bird(Animal):
        def __init__(self, pos=None):   
            if pos is None:
                edges = [
                    {'x': 0, 'y': random.randint(0, SCREEN_HEIGHT)},           
                    {'x': SCREEN_WIDTH, 'y': random.randint(0, SCREEN_HEIGHT)},
                    {'x': random.randint(0, SCREEN_WIDTH), 'y': 0},           
                    {'x': random.randint(0, SCREEN_WIDTH), 'y': SCREEN_HEIGHT}
                ]
                pos = random.choice(edges)
            super(Bird,self).__init__(pos, Bird_Size, (Bird_MIN_Speed + Bird_MAX_Speed) / 2) 
        
        def apply_force(self, boids):
            separation_force = pygame.math.Vector2(0, 0) #分離推力
            alignment_force = pygame.math.Vector2(0, 0) #對齊力
            cohesion_force = pygame.math.Vector2(0, 0) #聚集力
            center_of_mass = pygame.math.Vector2(0, 0) #聚集中心


            neighbor_count = 0 #紀錄偵測到的近鄰數量

            #遍歷其他的 boids
            for i in np.random.choice(np.arange(0, Bird_Number), size = (int(Bird_Number * Movement_Accuracy), ), replace = False):
                other = boids[i]
                #確保不是自己且對象是存活的
                if other is not self and other.situation == "alive":
                    #計算兩個 boid 之間的距離
                    distance = self.position.distance_to(other.position)

                    #檢查距離是否在排斥範圍內
                    if 0 < distance < Bird_Perception_Radius:
                        #計算推離力
                        separation_force += (self.position - other.position).normalize() * (Bird_MAX_Speed / distance)
                        #計算對齊力
                        alignment_force += other.velocity
                        #計算聚集中心
                        center_of_mass += other.position

                        neighbor_count += 1

            #總結
            if neighbor_count > 0:
                #計算推離力
                if separation_force.length_squared() > 0:
                    separation_force /= neighbor_count
                    if separation_force.length_squared() > Bird_Separation_Weight**2:
                        separation_force.scale_to_length(Bird_Separation_Weight)

                #計算對齊力
                #對齊力 = 理想速度 (平均速度 * 最大速度) - 目前速度
                if alignment_force.length_squared() > 0:
                    alignment_force /= neighbor_count
                    alignment_force = alignment_force.normalize() * Bird_MAX_Speed - self.velocity
                    if alignment_force.length_squared() > Bird_Alignment_Weight**2:
                        alignment_force.scale_to_length(Bird_Alignment_Weight)

                #計算聚集力
                #往質量中心移動
                center_of_mass /= neighbor_count
                cohesion_force = center_of_mass - self.position
                if cohesion_force.length_squared() > 0:
                    cohesion_force = cohesion_force.normalize() * Bird_MAX_Speed - self.velocity
                    if cohesion_force.length_squared() > Bird_Cohesion_Weight**2:
                        cohesion_force.scale_to_length(Bird_Cohesion_Weight)

            return separation_force + alignment_force + cohesion_force 

        def flee_predator(self, predators):
            flee_force = pygame.math.Vector2(0, 0)
            neighbor_count = 0

            for predator in predators:
                distance_vector = self.position - predator.position #從掠食者指向 boid 的向量
                distance = distance_vector.length()

                #檢查是否在掠食者的捕獲範圍內
                if distance < Predator_Eat_Radius:
                    self.situation = "dying"
                    self.speed = 0
                    #死亡後觸發粒子
                    pos = {'x': self.position.x, 'y': self.position.y}
                    particles.extend([Particle(pos,self.direction) for _ in range(Particle_Num)])

                #檢查是否在逃跑範圍內
                if distance < Bird_Alert_Radius and distance > 0:
                    #施加一個強大的推力
                    flee_force += distance_vector.normalize() * Bird_MAX_Speed / distance
                    neighbor_count += 1

            if neighbor_count > 0:
                #計算逃跑力
                if flee_force.length_squared() > 0:
                    flee_force /= neighbor_count
                    if flee_force.length_squared() > Bird_Flee_Weight**2:
                        flee_force.scale_to_length(Bird_Flee_Weight)
            return flee_force

        #當滑鼠左鍵按下時，附近的鳥群要遠離鼠標
        def mouse_activity(self):
            mouse_force = pygame.math.Vector2(0, 0) # boid 和滑鼠鼠標的斥力或引力
            if (Bird_Mouse_Activity["click"]!=0): # 若滑鼠沒點擊(或左右鍵一起點)則不用計算
                #邏輯和 flee_predator 類似
                #左鍵(Bird_Mouse_Activity["click"]=1)排斥，右鍵(Bird_Mouse_Activity["click"]=-1)吸引
                distance_vector = Bird_Mouse_Activity["click"]*(
                    self.position - pygame.Vector2(Bird_Mouse_Activity["pos"][0], Bird_Mouse_Activity["pos"][1]))
                distance = distance_vector.length()
                if 0 < distance < Bird_Alert_Radius:
                    mouse_force = distance_vector.normalize()*Bird_MAX_Speed / distance
                    if mouse_force.length_squared() > Bird_Flee_Weight**2:
                        mouse_force.scale_to_length(Bird_Flee_Weight)
            return mouse_force

        def update(self, all_boids, obstacles, predators):         
            if (self.situation == "alive"):
                #調整速度
                force = self.apply_force(all_boids) + self.flee_predator(predators) + self.mouse_activity() #計算作用力
                self.direction = (self.direction + force).normalize() #調整方向
                self.speed += force.length() #調整速率

                #實際運動
                super(Bird,self).basis_update(Bird_MAX_Speed, Bird_MIN_Speed, obstacles)

                #更改顏色/大小    
                self.color = Bird_Color_Slow+Bird_Color_ChangeRate * ((self.speed-Bird_MIN_Speed) / (Bird_MAX_Speed-Bird_MIN_Speed))
                self.size = Bird_Size
            elif (self.situation == "dying"):
                #死亡後:本體顏色漸暗
                self.color = (self.color[0] * 0.97, self.color[1] * 0.97, self.color[2] * 0.97)
                if (self.color[0] <= 15):
                    self.situation = "dead"


    class Predator(Animal):
        def __init__(self, pos = None):
            if pos is None:
                edges = [
                    {'x': 0, 'y': random.randint(0, SCREEN_HEIGHT)},          
                    {'x': SCREEN_WIDTH, 'y': random.randint(0, SCREEN_HEIGHT)},
                    {'x': random.randint(0, SCREEN_WIDTH), 'y': 0},           
                    {'x': random.randint(0, SCREEN_WIDTH), 'y': SCREEN_HEIGHT}
                ]
                pos = random.choice(edges)
            super(Predator, self).__init__(pos, Predator_Size, Predator_MAX_Speed, color=(255, 30, 45))
        
        #追蹤 bird
        def apply_track(self, all_boids):
            track_force = pygame.math.Vector2(0, 0) #追蹤力
            TRACK_RADIUS_SQ = Predator_Perception_Radius ** 2
            if all_boids:
                neighbor_count = 0 # 紀錄偵測到的近鄰數量

                for bird in all_boids:
                    if bird.situation != "alive":
                        continue
                    forward_bird = bird.position - self.position
                    if TRACK_RADIUS_SQ > forward_bird.length_squared():
                        track_force += forward_bird
                        neighbor_count += 1

                if neighbor_count > 0:
                    track_force /= neighbor_count
                    if track_force.length_squared() > 0:
                        track_force = track_force.normalize() * Predator_MAX_Speed - self.velocity
                        if track_force.length_squared() > Predator_Track_Weight ** 2:
                            track_force.scale_to_length(Predator_Track_Weight)

            return track_force
        
        #避開其他 predator
        def apply_separation(self, predators):
            separation_force = pygame.math.Vector2(0, 0)
            neighbor_count = 0

            for predator in predators:
                distance_vector = self.position - predator.position
                distance = distance_vector.length()

                #檢查是否在觀察範圍內
                if distance < Predator_Perception_Radius and distance > 0:
                    #施加推力
                    separation_force += distance_vector.normalize() * Predator_Perception_Radius / distance
                    neighbor_count += 1

            if neighbor_count > 0:
                #計算分離力
                if separation_force.length_squared() > 0:
                    separation_force /= neighbor_count
                    if separation_force.length_squared() > Predator_Separation_Weight**2:
                        separation_force.scale_to_length(Predator_Separation_Weight)

            return separation_force
        
        def update(self, all_boids, obstacles, predators):
            force = self.apply_track(all_boids)+self.apply_separation(predators) #計算作用力
            self.direction = (self.direction+force).normalize() #調整方向
            self.speed += force.length() #調整速率

            #調整大小
            self.size = Predator_Size

            #實際運動
            super(Predator, self).basis_update(Predator_MAX_Speed, Predator_MIN_Speed, obstacles)


    class Particle(Animal):
        def __init__(self, pos, dir):
            self.speed = random.randint(Particle_MIN_Speed, Particle_MAX_Speed)
            super(Particle,self).__init__(pos, Particle_Size, self.speed)
            self.lifetime = Particle_Lifetime
            self.direction = dir
            self.direction.rotate_ip(random.randint(-Particle_offset_angle, Particle_offset_angle))

        def update(self):
            self.velocity = self.direction*self.speed*DT
            self.position += self.velocity
            self.direction.rotate_ip(random.uniform(-20, 20)) #隨機擾動改變角度
            self.lifetime -= 1


        def draw(self, screen):
            #繪製圓形
            pygame.draw.circle(screen, self.color, (self.position[0], self.position[1]), Particle_Radious)


    class Obstacle:
        def __init__(self, vertices, color=(150, 150, 150)):
            #將頂點列表轉換為 Vector2 列表，方便運算
            self.vertices = [pygame.math.Vector2(v) for v in vertices]
            self.color = color

            #計算出多邊形的中心點
            if self.vertices:
                self.center = sum(self.vertices, pygame.math.Vector2(0, 0)) / len(self.vertices)
            else:
                self.center = pygame.math.Vector2(0, 0)

            self.radius = 0
            for vertix in self.vertices:
                self.radius = max(self.radius, (vertix - self.center).length())

        def draw(self, screen):
            #繪製實心多邊形
            int_vertices = [(int(v.x), int(v.y)) for v in self.vertices]
            if len(int_vertices) >= 3:
                pygame.draw.polygon(screen, self.color, int_vertices)

        def check_collision(self, boid) -> bool:
            """
            圓形碰撞
            極度簡化的碰撞邏輯，效能極高
            """ 
            #boid 的碰撞半徑
            COLLISION_RADIUS = boid.size

            #1. 計算 boid 到圓心的向量和距離
            center_to_boid = boid.position - self.center
            distance_sq = center_to_boid.length_squared()

            #總碰撞半徑 (兩者半徑之和) 的平方
            total_radius = self.radius + COLLISION_RADIUS
            total_radius_sq = total_radius ** 2

            #2. 檢查是否發生碰撞 (使用平方比較，避免 math.sqrt)
            if distance_sq < total_radius_sq:
                #距離
                distance = np.sqrt(distance_sq)

                #法線 (從圓心指向 boid 的向量)
                normal = center_to_boid

                if distance > 0: # 避免 boid 剛好在圓心
                    #推回位置：解決穿透
                    penetration_depth = total_radius - distance
                    normal_norm = normal / distance # 這裡的 normal / distance 就是正規化

                    #將 boid 沿著法線方向推開
                    boid.position += normal_norm * penetration_depth

                    #使用法線進行反射
                    boid.direction = boid.direction.reflect(normal_norm)
                    boid.speed *= Bounce_Damping

                    return True 

            return False
        @staticmethod
        def generate_random_polygon(center_x: float, center_y: float, min_radius: float, max_radius: float, num_vertices: int) -> list[tuple[float, float]]:
            """
            生成多邊形
            """
            center = pygame.math.Vector2(center_x, center_y)

            #1. 隨機生成 num_vertices 個半徑 (使用 NumPy)
            rand_radii = np.random.uniform(min_radius, max_radius, size = num_vertices)

            #2. 隨機生成角度
            angle_step = 2 * np.pi / num_vertices
            #基準角度：讓頂點均勻分佈在圓周上
            base_angles = np.arange(num_vertices) * angle_step

            #3. 隨機角度微調 (Jitter)：在 [-angle_step * 0.2, angle_step * 0.2] 範圍內
            angle_jitter = np.random.uniform(-angle_step * 0.2, angle_step * 0.2, size = num_vertices)
            final_angles = base_angles + angle_jitter

            #4. 向量化計算所有頂點座標
            x_coords = center.x + rand_radii * np.cos(final_angles)
            y_coords = center.y + rand_radii * np.sin(final_angles)

            #5. 組合成 [(x, y)] 列表
            vertices = []
            #使用 zip 將 x, y 座標打包
            for x, y in zip(x_coords, y_coords):
                vertices.append((x, y))

            return vertices

    # main
    # load
    birds = [Bird() for _ in range(Bird_Number)] #在邊緣生成 bird
    predators = [Predator() for _ in range(Predator_Number)]
    particles = []
    obstacles = [Obstacle(Obstacle.generate_random_polygon(
        random.randint(100, SCREEN_WIDTH - 100),
        random.randint(100, SCREEN_HEIGHT - 100),
        Obstacle_Size, int(Obstacle_Size * 1.4), random.randint(4, 20))) for _ in range(Obstacle_Number)]

    # tick
    while not stop_event.is_set():
        #取得數值
        Damping = shared_state["Overall"]["Damping"]/1000
        Movement_Accuracy = shared_state['Bird']['Movement_Accuracy']/100
        Bounce_Damping = shared_state['Overall']['Bounce_Damping']

        Bird_Number = shared_state['Bird']['Number']
        Bird_Size = shared_state['Bird']['Size']
        Bird_MIN_Speed = shared_state['Bird']['MIN_Speed']
        Bird_MAX_Speed = int(Bird_MIN_Speed*shared_state['Bird']['MAX_Speed_Multiplier'])
        Bird_Perception_Radius = shared_state['Bird']['Perception_Radius']
        Bird_Separation_Weight = shared_state['Bird']['Separation_Weight']
        Bird_Alignment_Weight = shared_state['Bird']['Alignment_Weight']
        Bird_Cohesion_Weight = shared_state['Bird']['Cohesion_Weight']
        Bird_Alert_Radius = shared_state['Bird']['Alert_Radius']
        Bird_Flee_Weight = shared_state['Bird']['Flee_Weight']

        Predator_Number = shared_state['Predator']['Number']
        Predator_Eat_Radius = shared_state['Predator']['Eat_Radius']
        Predator_Size = shared_state['Predator']['Size']
        Predator_MIN_Speed = shared_state['Predator']['MIN_Speed']
        Predator_MAX_Speed = int(Predator_MIN_Speed*shared_state['Predator']['MAX_Speed_Multiplier'])
        Predator_Perception_Radius = shared_state['Predator']['Perception_Radius']
        Predator_Track_Weight = shared_state['Predator']['Track_Weight']
        Predator_Separation_Weight = shared_state['Predator']['Separation_Weight']

        Obstacle_Number = shared_state['Obstacle']['Number']
        Obstacle_Size = shared_state['Obstacle']['Size']


        #取得動作
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                stop_event.set()
                break
        
        #Bird 數量更新
        desired_Bird_count = int(shared_state['Bird'].get("Number", len(birds)))
        if desired_Bird_count > len(birds):
            for _ in range(desired_Bird_count - len(birds)):
               birds.append(Bird())
        elif desired_Bird_count < len(birds):
               birds = birds[:desired_Bird_count]

        #掠食者數量更新
        desired_pred_count = int(shared_state['Predator'].get("Number", len(predators)))
        if desired_pred_count > len(predators):
            for _ in range(desired_pred_count - len(predators)):
                predators.append(Predator())
        elif desired_pred_count < len(predators):
            predators = predators[:desired_pred_count]

        #障礙物數量更新
        desired_obs_count = int(shared_state['Obstacle'].get("Number", len(obstacles)))
        if desired_obs_count > len(obstacles):
            for _ in range(desired_obs_count - len(obstacles)):
                size = int(shared_state['Obstacle'].get("Size", Obstacle_Size))
                obs = Obstacle(Obstacle.generate_random_polygon(
                    int(np.random.randint(100, SCREEN_WIDTH - 100)),
                    int(np.random.randint(100, SCREEN_HEIGHT - 100)),
                    size, int(size * 1.4), int(np.random.randint(4, 20))
                ))
                obstacles.append(obs)
        elif desired_obs_count < len(obstacles):
            obstacles = obstacles[:desired_obs_count]


        #繪圖
        Screen.fill(BACKGROUND_COLOR)
        
        #滑鼠事件處理
        Bird_Mouse_Activity["pos"] = pygame.mouse.get_pos() #滑鼠位置
        #滑鼠點擊處理，點左鍵為1(驅離)，點右鍵為-1(聚集)
        Bird_Mouse_Activity["click"] = pygame.mouse.get_pressed()[0]-pygame.mouse.get_pressed()[2]
        
        #更新死掉 bird 的狀態 
        for i in range(Bird_Number):
            #死掉的 bird 在邊界重生
            if (birds[i].situation == "dead"):
                birds[i] = Bird()
            else:
                birds[i].update(birds,obstacles,predators) 
                birds[i].draw(Screen)
        particles = [particle for particle in particles if (particle.lifetime > 0)] #移除週期結束的粒子
        for particle in particles:
            particle.update()
            particle.draw(Screen)
        for predator in predators:
            predator.update(birds, obstacles, predators)
            predator.draw(Screen)
        for obstacle in obstacles:
            obstacle.draw(Screen)

        pygame.display.flip()

        #計算 dt
        DT = Timer.tick(shared_state['Overall']['FPS']) / 1000
        shared_state['Overall']['DT'] = DT


    #結束清理
    pygame.quit()


# tkinter
def start_tkinter():
    #設定
    root = ttk.Window(themename = "superhero")
    root.title('Setting')
    root.geometry('540x360')

    #全域變數
    vars_dict = {
        "Overall":{
            "DT": ttk.IntVar(value = 0), #畫面更新率
            "FPS": ttk.IntVar(value = 60), #每秒幀數
            "Bounce_Damping": ttk.DoubleVar(value = 0.8), # bird 碰撞時能量遞減
            "Damping" : ttk.IntVar(value = 2) #阻力
        },
        "Bird": {
            "Number": ttk.IntVar(value = 300), # bird 數量
            "Size": ttk.IntVar(value = 8), # bird 大小
            "MIN_Speed": ttk.IntVar(value = 20), # bird 最小速度
            "MAX_Speed_Multiplier": ttk.DoubleVar(value = 10.0), # bird 最大速度
            "Perception_Radius": ttk.IntVar(value = 30), # bird 觀察範圍
            "Separation_Weight": ttk.DoubleVar(value = 1), #b ird 分離力最大值
            "Alignment_Weight": ttk.DoubleVar(value = 1), # bird 對齊力最大值
            "Cohesion_Weight": ttk.DoubleVar(value = 1), # bird 聚集力最大值
            "Flee_Weight": ttk.DoubleVar(value = 4), # bird 逃跑力最大值
            "Alert_Radius": ttk.IntVar(value = 50), # bird 警戒範圍

            #計算精度，若有 n 隻 bird ，則每隻 bird 需要與 n-1 隻 bird 互動，
            #為提升效能我這裡只讓 bird 與隨機 (n-1) * Movement_Accuracy 隻 bird 互動
            #另一種想法是讓每隻 bird 有 1 - Movement_Accuracy 的機率"不合群"，違背自然法則，模擬自然的隨機性
            "Movement_Accuracy": ttk.IntVar(value = 50), # bird 不合群率
        },
        "Predator": {
            "Number": ttk.IntVar(value = 3), # predator 數量
            "Size": ttk.IntVar(value = 10), # predator 大小
            "MIN_Speed": ttk.IntVar(value = 60), # predator 最小速度
            "MAX_Speed_Multiplier": ttk.DoubleVar(value = 3), # predator 最大速度
            "Perception_Radius": ttk.IntVar(value = 60), # predator 觀察範圍
            "Separation_Weight": ttk.DoubleVar(value = 1), # predator 分離力
            "Track_Weight": ttk.DoubleVar(value = 2), # predator 追蹤力
            "Eat_Radius": ttk.IntVar(value = 8), # predator 捕食範圍
        },
        "Obstacle": {
            "Number": ttk.IntVar(value = 4), # obstacle 數量
            "Size": ttk.IntVar(value = 80), # obstacle 大小
        }
    }

    shared_state = {
        "Overall": {k:vars_dict["Overall"][k].get() for k in vars_dict["Overall"]},
        "Bird": {k:vars_dict["Bird"][k].get() for k in vars_dict["Bird"]},
        "Predator": {k:vars_dict["Predator"][k].get() for k in vars_dict["Predator"]},
        "Obstacle": {k:vars_dict["Obstacle"][k].get() for k in vars_dict["Obstacle"]},
    }
    state_lock = threading.Lock()

    pygame_thread = threading.Thread(target=run_pygame, args = (shared_state, state_lock, stop_event))
    
    #切換畫面
    def switch_to(frame):
        frame.lift()

    def get_FPS():
        dt = shared_state["Overall"]["DT"]
        if dt > 0:
            fps = 1 / dt
            val_label.config(text = f"FPS: {fps:.2f}")
        root.after(100, get_FPS)
    
    #建立頁面
    topbar = ttk.Frame(root, bootstyle = "dark") #置頂選單區域
    topbar.pack(side = "top", fill = "x")

    right_topbar = ttk.Frame(topbar)
    right_topbar.pack(side = "right")

    content = ttk.Frame(root) #內容區域
    content.pack(fill = "both", expand = True)

    setting_bird = ttk.Frame(content)
    setting_predator = ttk.Frame(content)
    setting_obstacle = ttk.Frame(content)
    setting_overall = ttk.Frame(content)

    for frame in (setting_bird, setting_predator, setting_obstacle, setting_overall):
        frame.place(relx = 0, rely = 0, relwidth = 1, relheight = 1)

    for text, frame in [("Bird", setting_bird), ("Predator", setting_predator), ("Obstacle", setting_obstacle), ("Overall", setting_overall)]:
        ttk.Button(topbar, text = text, bootstyle = "outline-light", command = lambda f = frame: switch_to(f)).pack(side = "left", padx = 5, pady = 5)

    val_label = ttk.Label(right_topbar, text = "FPS: 0", width = 10)
    val_label.pack(side = "right", padx = 5)
    root.after(100, get_FPS)

    #垂直滾動區域
    def create_scrollable_frame(parent):
        canvas = ttk.Canvas(parent)
        scrollbar = ttk.Scrollbar(parent, orient = "vertical", command = canvas.yview)
        canvas.configure(yscrollcommand = scrollbar.set)

        scrollbar.pack(side = "right", fill = "y")
        canvas.pack(side = "left", fill = "both", expand = True)

        scrollable_frame = ttk.Frame(canvas)
        canvas.create_window((0, 0), window=scrollable_frame, anchor = "nw")

        #更新捲動區域大小
        def on_frame_configure(event):
            canvas.configure(scrollregion = canvas.bbox("all"))
        scrollable_frame.bind("<Configure>", on_frame_configure)

        #滾輪控制，只在 Canvas 上生效
        def on_mousewheel(event):
            delta = 0
            #W indows/Mac
            if hasattr(event, "delta"):
                try:
                    delta = int(-1 * (event.delta / 120))
                except Exception:
                    delta = 0
            else:
                # Linux
                if getattr(event, "num", None) == 4:
                    delta = -1
                elif getattr(event, "num", None) == 5:
                    delta = 1

            if delta:
                canvas.yview_scroll(delta, "units")

        def _bind_mousewheel(event):
            canvas.bind_all("<MouseWheel>", on_mousewheel)
            canvas.bind_all("<Button-4>", on_mousewheel)
            canvas.bind_all("<Button-5>", on_mousewheel)

        def _unbind_mousewheel(event):
            canvas.unbind_all("<MouseWheel>")
            canvas.unbind_all("<Button-4>")
            canvas.unbind_all("<Button-5>")

        canvas.bind("<Enter>", _bind_mousewheel) #滑鼠進入時綁定
        canvas.bind("<Leave>", _unbind_mousewheel) #離開時解除綁定

        return scrollable_frame, canvas

    #建立滑桿函數
    def add_slider(parent, label_text, var, from_, to_, step = 1.0, section = None):
        row_frame = ttk.Frame(parent)
        row_frame.pack(fill = "x", pady = 8, padx = 10)

        #標籤
        ttk.Label(row_frame, text = label_text, width = 20).pack(side = "left", anchor = "w")

        #數值
        val_label = ttk.Label(row_frame, text = str(var.get()), width = 6, anchor = "e")
        val_label.pack(side = "left", padx = 5)

        def update_label(v):
            if isinstance(var, ttk.DoubleVar):
                val_label.config(text = f"{float(v):.1f}")
            else:
                val_label.config(text = str(int(float(v))))

        #更新數值
        def adjust(v):
            #轉型並限制範圍
            try:
                new_val = var.get() + v
                if new_val < from_:
                    new_val = from_
                elif new_val > to_:
                    new_val = to_
                var.set(round(new_val, 2))
                update_label(new_val)
            except ValueError:
                #忽略無效的轉型請求
                pass

        #按鈕滑桿
        ttk.Button(row_frame, text = "-", bootstyle = "secondary", width = 1, command = lambda: adjust(-step)).pack(side = "left", padx = 2)
        slider = ttk.Scale(
            row_frame,  from_ = from_, to = to_, orient = "horizontal", length = 200, variable = var, command = update_label)
        slider.pack(side = "left", padx = 8, expand = True)
        ttk.Button(row_frame, text = "+", bootstyle = "secondary", width = 1, command = lambda: adjust(step)).pack(side = "left", padx = 2)
    

    # ======== overall ========
    Overall_scrollable_frame, overall_canvas = create_scrollable_frame(setting_overall)

    add_slider(Overall_scrollable_frame, "Bounce Damping", vars_dict["Overall"]["Bounce_Damping"], 0, 10, step = 0.1, section = "Overall")
    add_slider(Overall_scrollable_frame, "Damping(0.001)", vars_dict["Overall"]["Damping"], 0, 100, step = 1, section = "Overall")
    
    # ======== 鳥群 ========
    bird_scrollable_frame, bird_canvas = create_scrollable_frame(setting_bird)  

    add_slider(bird_scrollable_frame, "Bird Number", vars_dict["Bird"]["Number"], 0, 1000, step = 1, section = "Bird")
    add_slider(bird_scrollable_frame, "Bird Size", vars_dict["Bird"]["Size"], 1, 100, step = 1, section = "Bird")
    add_slider(bird_scrollable_frame, "Min Speed", vars_dict["Bird"]["MIN_Speed"], 1, 400, step = 1, section = "Bird")
    add_slider(bird_scrollable_frame, "Max Speed Multiplier", vars_dict["Bird"]["MAX_Speed_Multiplier"], 1, 20, step = 0.1, section = "Bird")
    add_slider(bird_scrollable_frame, "Perception Radius", vars_dict["Bird"]["Perception_Radius"], 0, 200, step = 1, section = "Bird")
    add_slider(bird_scrollable_frame, "Separation Weight", vars_dict["Bird"]["Separation_Weight"], 0, 20, step = 0.1, section = "Bird")
    add_slider(bird_scrollable_frame, "Alignment Weight", vars_dict["Bird"]["Alignment_Weight"], 0, 20, step = 0.1, section = "Bird")
    add_slider(bird_scrollable_frame, "Cohesion Weight", vars_dict["Bird"]["Cohesion_Weight"], 0, 20, step = 0.1, section = "Bird")
    add_slider(bird_scrollable_frame, "Flee Weight", vars_dict["Bird"]["Flee_Weight"], 0, 20, step = 0.1, section = "Bird")
    add_slider(bird_scrollable_frame, "Alert Radius", vars_dict["Bird"]["Alert_Radius"], 0, 200, step = 1, section = "Bird")
    add_slider(bird_scrollable_frame, "Movement Accuracy(%)", vars_dict["Bird"]["Movement_Accuracy"], 0, 100, step = 1, section = "Bird")

    # ======== 掠食者 ========
    predator_scrollable_frame, predator_canvas = create_scrollable_frame(setting_predator)

    add_slider(predator_scrollable_frame, "Predator Number", vars_dict["Predator"]["Number"], 0, 50, step = 1, section = "Predator")
    add_slider(predator_scrollable_frame, "Predator Size", vars_dict["Predator"]["Size"], 1, 100, step = 1, section = "Predator")
    add_slider(predator_scrollable_frame, "Min Speed", vars_dict["Predator"]["MIN_Speed"], 0, 400, step = 1, section = "Predator")
    add_slider(predator_scrollable_frame, "Max Speed Multiplier", vars_dict["Predator"]["MAX_Speed_Multiplier"], 1, 20,step = 0.1, section = "Predator")
    add_slider(predator_scrollable_frame, "Perception Radius", vars_dict["Predator"]["Perception_Radius"], 0, 200, step = 1, section = "Predator")
    add_slider(predator_scrollable_frame, "Separation Weight", vars_dict["Predator"]["Separation_Weight"], 0, 20, step = 0.1, section = "Predator")
    add_slider(predator_scrollable_frame, "Track Weight", vars_dict["Predator"]["Track_Weight"], 0, 20, step = 0.1, section = "Predator")
    add_slider(predator_scrollable_frame, "Eat Radius", vars_dict["Predator"]["Eat_Radius"], 0, 100, step = 1, section = "Predator")
    
    # ======== 障礙物 ========
    obstacle_scrollable_frame, obstacle_canvas = create_scrollable_frame(setting_obstacle)

    add_slider(obstacle_scrollable_frame, "Obstacle Number", vars_dict["Obstacle"]["Number"], 0, 20, step = 1, section = "Obstacle")


    def on_closing():
        stop_event.set()
        root.destroy()
    
    def update_shared_state():
        for section, vars_in_section in vars_dict.items():
            for key, var in vars_in_section.items():
                shared_state[section][key] = var.get()
        root.after(100, update_shared_state)      
    
    def check_pygame_stop():
        if stop_event.is_set():
            root.destroy()   # pygame 已經退出，關掉 tkinter
        else:
            root.after(100, check_pygame_stop)
    
    root.protocol("WM_DELETE_WINDOW", on_closing)
    update_shared_state()
    switch_to(setting_bird)
    check_pygame_stop()
    pygame_thread.start()
    root.mainloop()
    
    stop_event.set()

if __name__ == "__main__":
    start_tkinter()