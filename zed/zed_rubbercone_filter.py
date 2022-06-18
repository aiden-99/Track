class zed:
    def __init__(self):
        self.filter_size = 20 # 가운데 픽셀보다 얼마나 더 크게 볼건지 (정사각형 한 변)
        self.xy_offset = 0.3 # 점 주위로 몇 m 안까지 하나의 객체로 볼건지 (군집화 할건지)

    def pick_xy(self, point_cloud, xy): # 차량 뒷 축 기준으로 해당 픽셀의 x z값 반환
        err, point_cloud_value = point_cloud.get_value(int(xy[0]), int(xy[1])) # zed sdk 설치해야 돌아감
        return point_cloud_value[2] + 0.7, point_cloud_value[0]
    
    def depth_filter(self, point_cloud, pixel_xy): # pixel_xy [[x, y], [x, y]] 일케 넣어주면 왼쪽당, 오른쪽당 한번씩만 돌리면 됨
        filtered_xy = []
        # 인식된 라바콘 하나당
        for p_xy in pixel_xy:
            region_xy = [] # 주위 픽셀까지 뎁스 정보를 다 저장
            surrounding_dot_count = []# 주변 점 개수
            
            # 사각형으로 픽셀 뎁스정보 다 받아오기
            for i in range(len(self.filter_size)):
                for j in range(len(self.filter_size)):
                    try: # 해상도를 넘어서 뎁스 가져오는거는 그냥 패스
                        x, y = self.pick_xy(point_cloud, [p_xy[0] - self.filter_size/2 + j, p_xy[1]  - self.filter_size/2 + i])
                        region_xy.append([x, y])
                    except:
                        pass
            
            # 가장 주위에 많은 점 찾기
            for ind in range(len(region_xy)):
                dot_count = 0
                for every in region_xy:
                    # 주위에 있는지
                    if abs(region_xy[ind][0] - every[0]) < self.xy_offset and abs(region_xy[ind][1] - every[1]) < self.xy_offset:
                        dot_count += 1
                surrounding_dot_count.append(dot_count)

            temp = max(surrounding_dot_count)
            center_ind = surrounding_dot_count.index(temp)

            x_temp = []
            y_temp = []
            for every in region_xy:
                # 주위에 있는것들 평균
                if abs(region_xy[center_ind][0] - every[0]) < self.xy_offset and abs(region_xy[center_ind][1] - every[1]) < self.xy_offset:
                    x_temp.append(every[0])
                    y_temp.append(every[1])

            filtered_xy.append([np.mean(x_temp), np.mean(y_temp)])

        return filtered_xy
    
    def how_to_use(self):
        카메라 오픈, 파라미터 설정 하고 (욜로 이미지 돌릴때 미리 하겠지? 그냥 잊지 말라구 넣음)
        # Create a Camera object
        zed = sl.Camera()
        
        # Create a InitParameters object and set configuration parameters
        init_params = sl.InitParameters()
        init_params.depth_mode = sl.DEPTH_MODE.PERFORMANCE  # Use PERFORMANCE depth mode
        init_params.coordinate_units = sl.UNIT.METER  # Use meter units (for depth measurements)
        init_params.camera_resolution = sl.RESOLUTION.HD720

        # Open the camera
        zed.open(init_params)

        # Create and set RuntimeParameters after opening the camera
        runtime_parameters = sl.RuntimeParameters()
        runtime_parameters.sensing_mode = sl.SENSING_MODE.STANDARD  # Use STANDARD sensing mode
        # Setting the depth confidence parameters
        runtime_parameters.confidence_threshold = 100
        runtime_parameters.textureness_confidence_threshold = 100

        -----위에는 기존에 없으면 추가해줘야 하는거-----
        -----아래는 다 있어야 하는거---------
        point_cloud = sl.Mat()
        zed.retrieve_measure(point_cloud, sl.MEASURE.XYZRGBA)

        노랑 픽셀 센터 좌표 = [[x1, y1], [x2, y2]]
        파랑 픽셀 센터 좌표 = [[x1, y1], [x2, y2]]

        필터된 노랑 라바콘 xyz좌표 = self.depth_filter(point_cloud, 노랑 픽셀 센터 좌표)
        필터된 파랑 라바콘 xyz좌표 = self.depth_filter(point_cloud, 파랑 픽셀 센터 좌표)

