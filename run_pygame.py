import pygame
import numpy.random as random
import numpy as np

#setting
SCREEN_WIDTH = 1280
SCREEN_HEIGHT = 720
SCREEN_CNETER=(SCREEN_WIDTH/2,SCREEN_HEIGHT/2)
BACKGROUND_COLOR = (25, 25, 25)

def run_pygame(Setting, stop_event=None, shared_state_modify=None, shared_state_read=None):

    #初始化
    pygame.init()
    Screen = pygame.display.set_mode((SCREEN_WIDTH,SCREEN_HEIGHT))
    pygame.display.set_caption('boids V3.0.0')
    Timer = pygame.time.Clock()

    #全域變數
    Bird_Color_Slow = pygame.math.Vector3(*Setting["Overall_Bird"]["Color_Slow"]) #bird 最慢速顏色
    Bird_Color_Fast = pygame.math.Vector3(*Setting["Overall_Bird"]["Color_Fast"]) #bird 最快速顏色
    Bird_Color_ChangeRate = Bird_Color_Fast-Bird_Color_Slow
    Bird_Mouse_Activity = {"pos":(0,0),"click":0}

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
            self.speed *= (1 - Setting["Overall"]["Damping"])
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
            self.Attribute = Setting["Bird"].copy()
            super(Bird,self).__init__(pos, self.Attribute["Size"], (self.Attribute["MIN_Speed"] + self.Attribute["MAX_Speed"]) / 2) 
        
        def apply_force(self, boids, Target):
            separation_force = pygame.math.Vector2(0, 0) #分離推力
            alignment_force = pygame.math.Vector2(0, 0) #對齊力
            cohesion_force = pygame.math.Vector2(0, 0) #聚集力
            center_of_mass = pygame.math.Vector2(0, 0) #聚集中心


            neighbor_count = 0 #紀錄偵測到的近鄰數量

            #遍歷其他的 boids
            for i in Target:
                other = boids[i]
                #確保不是自己且對象是存活的
                if other is not self and other.situation == "alive":
                    #計算兩個 boid 之間的距離
                    distance = self.position.distance_to(other.position)

                    #檢查距離是否在排斥範圍內
                    if 0 < distance < self.Attribute["Perception_Radius"]:
                        #計算推離力
                        separation_force += (self.position - other.position).normalize() * (self.Attribute["MAX_Speed"] / distance)
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
                    if separation_force.length_squared() > self.Attribute["Separation_Weight"]**2:
                        separation_force.scale_to_length(self.Attribute["Separation_Weight"])

                #計算對齊力
                #對齊力 = 理想速度 (平均速度 * 最大速度) - 目前速度
                if alignment_force.length_squared() > 0:
                    alignment_force /= neighbor_count
                    alignment_force = alignment_force.normalize() * self.Attribute["MAX_Speed"] - self.velocity
                    if alignment_force.length_squared() > self.Attribute["Alignment_Weight"]**2:
                        alignment_force.scale_to_length(self.Attribute["Alignment_Weight"])

                #計算聚集力
                #往質量中心移動
                center_of_mass /= neighbor_count
                cohesion_force = center_of_mass - self.position
                if cohesion_force.length_squared() > 0:
                    cohesion_force = cohesion_force.normalize() * self.Attribute["MAX_Speed"] - self.velocity
                    if cohesion_force.length_squared() > self.Attribute["Cohesion_Weight"]**2:
                        cohesion_force.scale_to_length(self.Attribute["Cohesion_Weight"])

            return separation_force + alignment_force + cohesion_force 

        def flee_predator(self, predators):
            flee_force = pygame.math.Vector2(0, 0)
            neighbor_count = 0

            for predator in predators:
                distance_vector = self.position - predator.position #從掠食者指向 boid 的向量
                distance = distance_vector.length()

                #檢查是否在掠食者的捕獲範圍內
                if distance < predator.Attribute["Eat_Radius"]:
                    self.situation = "dying"
                    self.speed = 0
                    #死亡後觸發粒子
                    pos = {'x': self.position.x, 'y': self.position.y}
                    particles.extend([Particle(pos,self.direction) for _ in range(Setting["Particle"]["Num"])])

                #檢查是否在逃跑範圍內
                if distance < self.Attribute["Alert_Radius"] and distance > 0:
                    #施加一個強大的推力
                    flee_force += distance_vector.normalize() * self.Attribute["MAX_Speed"] / distance
                    neighbor_count += 1

            if neighbor_count > 0:
                #計算逃跑力
                if flee_force.length_squared() > 0:
                    flee_force /= neighbor_count
                    if flee_force.length_squared() > self.Attribute["Flee_Weight"]**2:
                        flee_force.scale_to_length(self.Attribute["Flee_Weight"])
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
                if 0 < distance < self.Attribute["Alert_Radius"]:
                    mouse_force = distance_vector.normalize()*self.Attribute["MAX_Speed"] / distance
                    if mouse_force.length_squared() > self.Attribute["Flee_Weight"]**2:
                        mouse_force.scale_to_length(self.Attribute["Flee_Weight"])
            return mouse_force

        def update(self, all_boids, obstacles, predators, Target):         
            if (self.situation == "alive"):
                #調整速度
                force = self.apply_force(all_boids, Target) + self.flee_predator(predators) + self.mouse_activity() #計算作用力
                self.direction = (self.direction + force).normalize() #調整方向
                self.speed += force.length() #調整速率

                #實際運動
                super(Bird,self).basis_update(self.Attribute["MAX_Speed"], self.Attribute["MIN_Speed"], obstacles)

                #更改顏色/大小
                OMxS = Setting["Overall_Bird"]["MAX_Speed"]
                OMnS = Setting["Overall_Bird"]["MIN_Speed"]
                self.color = Bird_Color_Slow+(Bird_Color_ChangeRate)*((self.speed-OMnS) / (OMxS-OMnS) if (OMxS-OMnS>0) else 1)
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
            self.Attribute = Setting["Predator"]
            super(Predator, self).__init__(pos, self.Attribute["Size"], self.Attribute["MAX_Speed"], color=(255, 30, 45))
        
        #追蹤 bird
        def apply_track(self, all_boids):
            track_force = pygame.math.Vector2(0, 0) #追蹤力
            TRACK_RADIUS_SQ = self.Attribute["Perception_Radius"] ** 2
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
                        track_force = track_force.normalize() * self.Attribute["MAX_Speed"] - self.velocity
                        if track_force.length_squared() > self.Attribute["Track_Weight"] ** 2:
                            track_force.scale_to_length(self.Attribute["Track_Weight"])

            return track_force
        
        #避開其他 predator
        def apply_separation(self, predators):
            separation_force = pygame.math.Vector2(0, 0)
            neighbor_count = 0

            for predator in predators:
                distance_vector = self.position - predator.position
                distance = distance_vector.length()

                #檢查是否在觀察範圍內
                if distance < self.Attribute["Perception_Radius"] and distance > 0:
                    #施加推力
                    separation_force += distance_vector.normalize() * self.Attribute["Perception_Radius"] / distance
                    neighbor_count += 1

            if neighbor_count > 0:
                #計算分離力
                if separation_force.length_squared() > 0:
                    separation_force /= neighbor_count
                    if separation_force.length_squared() > self.Attribute["Separation_Weight"]**2:
                        separation_force.scale_to_length(self.Attribute["Separation_Weight"])

            return separation_force
        
        def update(self, all_boids, obstacles, predators):
            force = self.apply_track(all_boids)+self.apply_separation(predators) #計算作用力
            self.direction = (self.direction+force).normalize() #調整方向
            self.speed += force.length() #調整速率

            #調整大小
            self.size = self.Attribute["Size"]

            #實際運動
            super(Predator, self).basis_update(self.Attribute["MAX_Speed"], self.Attribute["MIN_Speed"], obstacles)


    class Particle(Animal):
        def __init__(self, pos, dir):
            self.speed = random.randint(Setting["Particle"]["MIN_Speed"], Setting["Particle"]["MAX_Speed"])
            super(Particle,self).__init__(pos, Setting["Particle"]["Size"], self.speed)
            self.lifetime = Setting["Particle"]["Lifetime"]
            self.direction = dir
            self.direction.rotate_ip(random.randint(-Setting["Particle"]["offset_angle"], Setting["Particle"]["offset_angle"]))

        def update(self):
            self.velocity = self.direction*self.speed*DT
            self.position += self.velocity
            self.direction.rotate_ip(random.uniform(-20, 20)) #隨機擾動改變角度
            self.lifetime -= 1


        def draw(self, screen):
            #繪製圓形
            pygame.draw.circle(screen, self.color, (self.position[0], self.position[1]), Setting["Particle"]["Radious"])


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
                    boid.speed *= Setting["Overall"]["Bounce_Damping"]

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
    birds = [Bird() for _ in range(Setting["Overall_Bird"]["Number"])] #在邊緣生成 bird
    predators = [Predator() for _ in range(Setting["Overall_Predator"]["Number"])]
    particles = []
    obstacles = [Obstacle(Obstacle.generate_random_polygon(
        random.randint(100, SCREEN_WIDTH - 100),
        random.randint(100, SCREEN_HEIGHT - 100),
        Setting["Obstacle"]["Size"], int(Setting["Obstacle"]["Size"] * 1.4), random.randint(4, 20))) for _ in range(Setting["Obstacle"]["Number"])]

    # tick
    while not stop_event.is_set():
        #取得動作
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                stop_event.set()
                break
        
        #暫停
        if shared_state_modify and shared_state_read:
            if shared_state_read['Overall']['Pause']:
                Timer.tick(Setting["Overall"]["FPS"]) #處理時間
                continue

        #Bird 數量更新
        desired_Bird_count = int(Setting["Overall_Bird"]["Number"])
        if desired_Bird_count > len(birds):
            for _ in range(desired_Bird_count - len(birds)):
               birds.append(Bird())
        elif desired_Bird_count < len(birds):
               birds = birds[:desired_Bird_count]

        #掠食者數量更新
        desired_pred_count = int(Setting["Overall_Predator"]["Number"])
        if desired_pred_count > len(predators):
            for _ in range(desired_pred_count - len(predators)):
                predators.append(Predator())
        elif desired_pred_count < len(predators):
            predators = predators[:desired_pred_count]

        #障礙物數量更新
        desired_obs_count = int(Setting["Obstacle"]["Number"])
        if desired_obs_count > len(obstacles):
            for _ in range(desired_obs_count - len(obstacles)):
                size = int(Setting["Obstacle"]["Size"])
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
        
        #更新 bird 的狀態 
        for i in range(Setting["Overall_Bird"]["Number"]):
            #死掉的 bird 在邊界重生
            if (birds[i].situation == "dead"):
                birds[i] = Bird()
            else:
                birds[i].update(birds,obstacles,predators,
                    Target=np.random.choice(np.arange(0, Setting["Overall_Bird"]["Number"]), size = (int(Setting["Overall_Bird"]["Number"] * Setting["Overall_Bird"]["Movement_Accuracy"]), ), replace = False)
                ) 
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
        DT = Timer.tick(Setting['Overall']['FPS']) / 1000
        if shared_state_modify and shared_state_read: shared_state_modify['Overall']['DT'] = DT


    #結束清理
    pygame.quit()