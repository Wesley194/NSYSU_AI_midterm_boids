import ttkbootstrap as ttk
import run_pygame
import threading
import read_data

#load data
Pygame_Setting = read_data.read_Setting()

# tkinter
def set_tkinter(stop_event=threading.Event()):
    #設定
    root = ttk.Window(themename = "superhero")
    root.title('Main Console')
    root.geometry('580x640')

    #data
    vars_dict_modify = {
        "Overall":{
            "Bounce_Damping": ttk.DoubleVar(value = 0.8), # bird 碰撞時能量遞減
            "Damping" : ttk.IntVar(value = 2), #阻力
            "Pause" : ttk.BooleanVar(value = False) #是否暫停
        },
        "Overall_Bird": {
            "Number": ttk.IntVar(value = 300), # bird 數量
        },
        "Overall_Predator": {
            "Number": ttk.IntVar(value = 3), # predator 數量
        },
        "Predator": {
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

    vars_dict_read = {
        "Overall":{
            "DT": ttk.StringVar(value = ""), #畫面更新率
        },
        "Bird": {
            "Size": ttk.StringVar(value = ""), # bird 大小
            "MIN_Speed": ttk.StringVar(value = ""), # bird 最小速度
            "MAX_Speed": ttk.StringVar(value = ""), # bird 最大速度
            "Perception_Radius": ttk.StringVar(value = ""), # bird 觀察範圍
            "Separation_Weight": ttk.StringVar(value = ""), #bird 分離力最大值
            "Alignment_Weight": ttk.StringVar(value = ""), # bird 對齊力最大值
            "Cohesion_Weight": ttk.StringVar(value = ""), # bird 聚集力最大值
            "Flee_Weight": ttk.StringVar(value = ""), # bird 逃跑力最大值
            "Alert_Radius": ttk.StringVar(value = ""), # bird 警戒範圍
        }
    }

    # initial data
    for section, vars_in_section in vars_dict_modify.items():
            for key, var in vars_in_section.items():
                if section in Pygame_Setting and key in Pygame_Setting[section]:
                    var.set(Pygame_Setting[section][key])

    # handle shared state
    shared_state_modify = {
        title: {key: value.get() for key,value in params.items()}
        for title,params in vars_dict_modify.items()
    }
    shared_state_read = {
        title: {key: 0 for key,value in params.items()}
        for title,params in vars_dict_read.items()
    }
    
    #建立頁面
    topbar = ttk.Frame(root, bootstyle = "dark") #置頂選單區域
    topbar.pack(side = "top", fill = "x")

    right_topbar = ttk.Frame(topbar)
    right_topbar.pack(side = "right")

    content = ttk.Frame(root) #內容區域
    content.pack(fill = "both", expand = True)

    Console_Window = {
        "Console" : ttk.Frame(content),
        "Sim Set" : ttk.Frame(content),
        "Overlook" : ttk.Frame(content),
    }

    for frame in Console_Window.values():
        frame.place(relx = 0, rely = 0, relwidth = 1, relheight = 1)

    for text, frame in Console_Window.items():
        ttk.Button(topbar, text = text, bootstyle = "outline-light", command = lambda f = frame: switch_to(f)).pack(side = "left", padx = 5, pady = 5)

    val_label_FPS = ttk.Label(right_topbar, text = "FPS: 0", width = 10)
    val_label_FPS.pack(side = "right", padx = 5)
    pause_button = ttk.Button(right_topbar, text = "||", bootstyle = "outline-light", command = (lambda : switch_Pause())).pack(side = "right", padx = 5)

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
    
    def add_readonly_value(parent, label_text, section, key):
        row_frame = ttk.Frame(parent)
        row_frame.pack(fill="x", pady=8, padx=10)

        # 左邊文字標籤
        ttk.Label(row_frame, text=label_text, width=20).pack(side="left", anchor="w")

        # 顯示數值用的 Tk 變數
        val_var = vars_dict_read[section][key]

        # 顯示數值的 Label
        ttk.Label(row_frame, textvariable=val_var, width=10, anchor="e").pack(side="left", padx=5)

    def add_text_label(parent, label_text, font=None,color="#FFFFFF"):
        row_frame = ttk.Frame(parent)
        row_frame.pack(fill="x", pady=8, padx=10)
        ttk.Label(row_frame, text=label_text, width=20, font=font, foreground=color).pack(side="left", anchor="w")

    
    # ======== Console ========
    console_scrollable_frame, overall_canvas = create_scrollable_frame(Console_Window["Console"])


    # ======== 模擬設定 ========
    simSet_scrollable_frame, bird_canvas = create_scrollable_frame(Console_Window["Sim Set"])  

    add_text_label(simSet_scrollable_frame,"Overall",font=("Helvetica",14,"bold"),color="#EFA00B")
    add_slider(simSet_scrollable_frame, "Bounce Damping", vars_dict_modify["Overall"]["Bounce_Damping"], 0, 10, step = 0.1, section = "Overall")
    add_slider(simSet_scrollable_frame, "Damping(0.001)", vars_dict_modify["Overall"]["Damping"], 0, 100, step = 1, section = "Overall")
    
    add_text_label(simSet_scrollable_frame,"Predator",font=("Helvetica",14,"bold"),color="#EFA00B")
    add_slider(simSet_scrollable_frame, "Number", vars_dict_modify["Overall_Predator"]["Number"], 0, 50, step = 1, section = "Predator")
    add_slider(simSet_scrollable_frame, "Size", vars_dict_modify["Predator"]["Size"], 1, 100, step = 1, section = "Predator")
    add_slider(simSet_scrollable_frame, "Min Speed", vars_dict_modify["Predator"]["MIN_Speed"], 1, 400, step = 1, section = "Predator")
    add_slider(simSet_scrollable_frame, "Max Speed Multiplier", vars_dict_modify["Predator"]["MAX_Speed_Multiplier"], 1, 20,step = 0.1, section = "Predator")
    add_slider(simSet_scrollable_frame, "Perception Radius", vars_dict_modify["Predator"]["Perception_Radius"], 0, 200, step = 1, section = "Predator")
    add_slider(simSet_scrollable_frame, "Separation Weight", vars_dict_modify["Predator"]["Separation_Weight"], 0, 20, step = 0.1, section = "Predator")
    add_slider(simSet_scrollable_frame, "Track Weight", vars_dict_modify["Predator"]["Track_Weight"], 0, 20, step = 0.1, section = "Predator")
    add_slider(simSet_scrollable_frame, "Eat Radius", vars_dict_modify["Predator"]["Eat_Radius"], 0, 100, step = 1, section = "Predator")
    
    add_text_label(simSet_scrollable_frame,"Obstacle",font=("Helvetica",14,"bold"),color="#EFA00B")
    add_slider(simSet_scrollable_frame, "Obstacle Number", vars_dict_modify["Obstacle"]["Number"], 0, 20, step = 1, section = "Obstacle")

    # ======== 監看視窗 ========
    overlook_scrollable_frame, predator_canvas = create_scrollable_frame(Console_Window["Overlook"])
    add_text_label(overlook_scrollable_frame,"Bird",font=("Helvetica",14,"bold"),color="#EFA00B")
    add_readonly_value(overlook_scrollable_frame, "Size", "Bird", "Size")
    add_readonly_value(overlook_scrollable_frame, "MIN Speed", "Bird", "MIN_Speed")
    add_readonly_value(overlook_scrollable_frame, "MAX Speed", "Bird", "MAX_Speed")
    add_readonly_value(overlook_scrollable_frame, "Perception Radius", "Bird", "Perception_Radius")
    add_readonly_value(overlook_scrollable_frame, "Separation Weight", "Bird", "Separation_Weight")
    add_readonly_value(overlook_scrollable_frame, "Alignment Weight", "Bird", "Alignment_Weight")
    add_readonly_value(overlook_scrollable_frame, "Cohesion Weight", "Bird", "Cohesion_Weight")
    add_readonly_value(overlook_scrollable_frame, "Flee Weight", "Bird", "Flee_Weight")
    add_readonly_value(overlook_scrollable_frame, "Alert Radius", "Bird", "Alert_Radius")

    

    def on_closing():
        stop_event.set()
        root.destroy()
    
    def update_shared_state():
        Specialize = ("Damping","MAX_Speed_Multiplier","Movement_Accuracy")
        for section, vars_in_section in vars_dict_modify.items():
            for key, var in vars_in_section.items():
                val = var.get()
                if key in Specialize:
                    if key=="Damping":
                        Pygame_Setting["Overall"]["Damping"] = val/1000
                    elif section=="Predator" and key=="MAX_Speed_Multiplier":
                        Pygame_Setting["Predator"]["MAX_Speed"] = int(Pygame_Setting["Predator"]["MIN_Speed"]*val)
                    elif key=="Movement_Accuracy":
                        Pygame_Setting["Overall_Bird"]["Movement_Accuracy"] = val/100
                elif section not in Pygame_Setting or key not in Pygame_Setting[section]:
                    shared_state_modify[section][key] = val
                else :
                    Pygame_Setting[section][key] = val

        for section, vars_in_section in vars_dict_read.items():
            for key, var in vars_in_section.items():
                var.set(f"{float(shared_state_read[section][key]):.2f}")
        root.after(100, update_shared_state)      
    
    def check_pygame_stop():
        if stop_event.is_set():
            root.destroy()   # pygame 已經退出，關掉 tkinter
        else:
            root.after(100, check_pygame_stop)

    #切換畫面
    def switch_to(frame):
        frame.lift()

    def get_FPS(val_label=val_label_FPS):
        dt = shared_state_read["Overall"]["DT"]
        if dt > 0:
            fps = 1 / dt
            val_label.config(text = f"FPS: {fps:.2f}")
        root.after(100, get_FPS)
    
    def switch_Pause():
        vars_dict_modify["Overall"]["Pause"].set(not vars_dict_modify["Overall"]["Pause"].get())
        # shared_state_read["Overall"]["Pause"]=not shared_state_read["Overall"]["Pause"]
    
    root.protocol("WM_DELETE_WINDOW", on_closing)
    update_shared_state()
    switch_to(Console_Window["Console"])
    check_pygame_stop()
    get_FPS()
    pygame_thread = threading.Thread(target=run_pygame.run_pygame, args = (Pygame_Setting, stop_event, shared_state_read, shared_state_modify))
    pygame_thread.start()
    root.mainloop()
    
    stop_event.set()

if __name__ == "__main__":
    set_tkinter()