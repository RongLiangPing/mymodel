def compute_deg():
    # 读取图的连接关系文件
    graph_file_path = "graph.txt"
    with open(graph_file_path, 'r') as file:
        lines = file.readlines()

    # 统计每个顶点的度
    degrees = {}
    for line in lines:
        v1, v2 = map(int, line.strip().split())
        degrees[v1] = degrees.get(v1, 0) + 1
        degrees[v2] = degrees.get(v2, 0) + 1

    # 将度信息写入文件
    deg_file_path = "deg.txt"
    with open(deg_file_path, 'w') as file:
        for vertex, degree in degrees.items():
            file.write(f"{vertex} {degree}\n")
compute_deg()


def calculate_deg_matrix(deg_file_path, matrix_output_file):
    def read_deg_file(file_path):
        deg_data = {}
        with open(file_path, 'r') as file:
            for line in file:
                node, degree = map(int, line.strip().split())
                deg_data[node] = degree
        return deg_data

    def calculate_matrix(deg_data, output_file):
        n = len(deg_data)

        with open(output_file, 'w') as file:
            total_data = n * n  # 计算总数据量
            processed_data = 0  # 已处理数据量
            progress_counter = 0  # 进度计数器
            for i in range(1, n + 1):
                deg_vi = deg_data.get(i, 0)  # 获取节点i的度数，若不存在则默认为0
                for j in range(1, n + 1):
                    deg_vj = deg_data.get(j, 0)  # 获取节点j的度数，若不存在则默认为0
                    matrix_value = deg_vi / (deg_vi + deg_vj) if deg_vi + deg_vj != 0 else 0
                    file.write(f"{float(matrix_value):.6f} ")  # 直接将整数转换为浮点数并写入文件
                    
                    processed_data += 1  # 更新已处理数据量
                    progress_counter += 1  # 更新进度计数器
                    if progress_counter == 100000:  # 每处理十万条数据打印一次进度
                        progress = processed_data / total_data * 100  # 计算进度百分比
                        print(f"当前进度：{progress:.2f}%")  # 打印进度百分比
                        progress_counter = 0  # 重置进度计数器
            print("矩阵计算完成并已保存至 deg_matrix.txt 文件中。")

    deg_data = read_deg_file(deg_file_path)
    calculate_matrix(deg_data, matrix_output_file)

# 调用函数并传入文件路径参数
deg_file_path = "deg.txt"
matrix_output_file = "deg_matrix.txt"
compute_deg()
calculate_deg_matrix(deg_file_path, matrix_output_file)
