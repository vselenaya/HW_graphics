#ifdef WIN32
#include <SDL.h>
#undef main
#else
#include <SDL2/SDL.h>
#endif

#include <GL/glew.h>

#include <string_view>
#include <stdexcept>
#include <iostream>
#include <chrono>
#include <vector>
#include <map>


static int W = 400;  // размеры поля (сетки) для рисования графика: 
static int H = 400;  // это означает, что рисованием производим по сетке [0, W] на [0, H] делений (чем больше W и H, тем плавнее и точнее график)
#define lo 0.2  // наименьшее и наибольшее значение функции, которое мы рассматриваем
#define hi 1.5  // (между ниими рисуются изолинии и выбирается цвет графика)



// ====== Блок общих функций для вывода информации об ошибках ======
std::string to_string(std::string_view str) {
    return std::string(str.begin(), str.end());
}

void sdl2_fail(std::string_view message) {
    throw std::runtime_error(to_string(message) + SDL_GetError());
}

void glew_fail(std::string_view message, GLenum error) {
    throw std::runtime_error(to_string(message) + reinterpret_cast<const char *>(glewGetErrorString(error)));
}
// ======================



// Структура, отвечающая за цвет:
// цвет представляется как значения 4-ёх каналов: R(красный), G(зелёный), B(синий), A(прозрачность);
// значение в каждом канале - от 0 до 255 (один байт = 8 бит -> тип данныз uint8_t)
struct color {
    std::uint8_t R, G, B, A;
};


// Структура, отвечающая за кооррдинаты вершины на плоскости - просто две координаты x и y:
struct position {
    float x, y;
};


// Глобальный массив значений функции в каждом узле (пересечение вертикальных и горизонтальных делений) сетки;
// эти значения перечитываем каждый кадр
static float values_function[1000+1][1000+1];


// Функция для вычисления функции-гауссиана (трёхмерного) в точке (x,y) по формуле:
// f(x, y) = l * e^((x-x0)^2 / (2 sx^2) + (y-y0)^2 / (2 sy^2)), где
//           l - максимальная высота купола гауссиана, которая достиагется в его центре (x0, y0),
//           sx, sy - дисперсия по осям X и Y (отвечает за размашистость купола)
float gauss(float x, float y, float x0, float y0, float l, float sx, float sy) {
    return l * exp(-1 * (\
                         (x - x0) * (x - x0) / (2 * sx * sx) +\
                         (y - y0) * (y - y0) / (2 * sy * sy)\
                        ));
}


// Основная функция, проекцию которой на плоскость и будем рисовать по сетке:
// (здесь вычисляем её значение в целочисленной (= узел сетки) точке x,y во время t)
float function(int x, int y, float t) {
    return // какую-то смесь гауссианов, выглядящую симпатично удалось подобрать
    0.2 * gauss(x, y, 3*W/8.0, H/2.0, 2+10 * abs(sin(t/2)), W/4.f, H/8.f) +  // три гауссиана, купола которых: в середине, снизу-слева и сверху-справа плоскости
    0.2 * gauss(x, y, W/6.0, 5*H/6.0, 2+8 * abs(cos(t/3)), W/8.f, H/16.f) +
    0.2 * gauss(x, y, 7*W/8.0, H/6.0, 2+12 * abs(sin(t/4)), W/16.f, H/6.f) +
    0.2 * gauss(x, y, W * abs(sin(t)), H * abs(cos(t)), 6 , W/10.f, H/10.f) +  // гауссиан, который по дуге слева-снизу -> вправо-вврех бегает 
    0.2 * gauss(x, y, W * abs(sin(t/1.5)), H * abs(sin(t/1.5)), 6 , W/8.f, H/8.f);  // гауссиан, который по диагонали двигается слева-сверху вправо-вниз
}


// Функция, вычисляющая значения function на всех узлах сетки -> заполняем массив 
// (нужно вызывать в начале каждого кадра)
void sample_on_grid(float time) {
    for (int x = 0; x <= W; x ++)
        for (int y = 0; y <= H; y ++)
            values_function[x][y] = function(x, y, time);
}


// Функция, которая определяем (по значению функции), каким цветом раскрасить данную вершину сетки:
// возвращает указатель (точнее ссылку) на одномерный вектор, где сначала по строкам, затем по столбца хранятся
// все (W+1) * (H+1) цветов узлов сеточки
std::vector <color> &get_color() {
    static std::vector <color> samples;  // заводим массив (статический - то есть один раз создаётся за всю программу)
                                         // (точнее, один раз он создаётся внутри каждого потока, но программа у нас однопоточная, так что то же самое)
    samples.clear();  // очищаем с предыдущего шага                     

    for (int x = 0; x <= W; x ++) {
        for (int y = 0; y <= H; y ++) {
            float val = values_function[x][y];  // получаем значение функции в вершине (x, y) сетки
            float R, B;  // значения цветовых каналов, которые обозначат цвет данной вершины
            if (val > hi) {  // если значение большое - рисуем красным
                B = 0;
                R = 255;
            } else if (val < lo) {  // если маленькое - синим
                B = 255;
                R = 0;
            } else {  // в остальных случаях линейно интерполируем точку (R, B) на отрезке [(0, 255), (255, 0)] 
                float a = (1/(hi-lo) * val - lo/(hi-lo));
                R = 255 * a;
                B = 255 * (1 - a);
            }
            color col = {(uint8_t) R, 0, (uint8_t) B, 255};  // получаем цвет и 
            samples.push_back(col);  // добавили в список
        }
    }
    return samples;
} 



// Эта функция рассматривает отрезок [(x1, y1), (x2, y2)] (между вершинами сетки), на котором
// восстанавливает точку пересечения этого отрезка линией уровня function (линия уровня для значения = c -
// это множество точек, где function = c). Если такой точки ещё не было, то она добавляется в вектор
// points. Словарь prev хранит для каждого отрезка индекс (вннутри prev) точки (по нему мы можем узнать, былы
// ли такая точка найдена ранее (а такое могло быть, ведь могли функции от одного отрезка вызвать повторно) = 
// ребуется ли её добавлять в points). Вектор indices служит для экономии (чтобы потом по много раз не
// отрисовывать одни и те же точки) - он просто хранит индексы (внутри points) каждой восстановленной точки
// пересечения (вместо того, чтобы каждый раз добавлять точку в points, мы у повторяющейся точки запоминаем лишь
// индекс внутри points -> затем при отрисовке воспользуемся индексированным рендерингом, который отрисует
// только точки из points (а они уникальны, без дубликатов, поэтому зря не рисуем), но согласно порядку из indecis).
// (функция вовзращает true, если на отрезке нашлась точка пересечения и false иначе)
bool add_point_to_contour(int x1, int y1, int x2, int y2, float c,
                          std::vector <position> &points, std::vector <std::uint32_t> &indices,
                          std::map <std::tuple <int, int, int, int>, std::uint32_t> &prev) {
    float f1 = values_function[x1][y1];  // значения функции на концах отрезка
    float f2 = values_function[x2][y2];

    if ((f1 - c) * 1.L * (f2 - c) < 0) {  // если эти значения разные относительно константы 'c' (одно >, другое < или наоборот), 
                                          // то (считая функцию непрерывной) где-то внутри отрезка точно есть точка, где она = 'с' (это и есть искомое пересечение)
        if (prev.count({x1, y1, x2, y2}) == 0) {  // если для отрезка ещё неизвестна эта точка (отстуствует в словарике), то вычисляем:
            long double a = (f1 - c) * 1.L / (c - f2);  // любая точка на отрезке представляется как линейная комбинация: (x,y) = t * (x1,y1) + (1-t) * (x2,y2), где t \in [0,1] - параметр;
            float x = (a * x2 + x1) / (a + 1);          // так как отрезок короткий, будем считать нашу функцию f (которая в коде function) на отрезке линейной -> по линейности: f(x,y) = t * f(x1,y1) + (1-t) * f(x2,y2);
            float y = (a * y2 + y1) / (a + 1);          // уже знаем, что f(x1,y1) = f1, f(x2,y2) = f2, при этом (x,y) ищем такую, в которой f(x,y) = c -> подставляя такие значения в предыдущую формулу, получим t = (с-f2)/(f1-f2);
                                                        // а зная t, можем найти (x,y) = t * (x1,y1) + (1-t) * (x2,y2) -> получили точку! (в коде чуть сложнее - там используется a = t / (1-t)...)
            position new_point = {x, y};  // получили найденную точку
            points.push_back(new_point);  // так как её ещё не было, добавляем в вектор points
            prev[{x1, y1, x2, y2}] = points.size() - 1;  // запоминаем индекс точки в словаре для данного отрезка
            prev[{x2, y2, x1, y1}] = points.size() - 1;  // (так как отрезок можем рассматривать как [a,b] и как [b,a], то добавляем в сразу два элемента словаря)
        }

        indices.push_back(prev[{x1, y1, x2, y2}]);  // теперь, когда индекс точки на данном отрезке точно известен, добавляем его в вектор
        return true;  // возвращаем, что точка существует
    }

    return false;  // если f1 и f2 одного знака относительно 'c', то считаем, что точки пересечения нет (то есть тут линия уровня не проходит)
}


// Эта функци получает линии уровня как набор точек (при отрисовке их с помощью паарметра GL_LINES получатся
// все линии уровня). Для этого заполняются вектора points и indices (которые описаны ранее).
// (num - это количество линий уровня)
void get_contour(std::vector <position> &points, std::vector <std::uint32_t> &indices, int num=1) {
    static std::map <std::tuple <int, int, int, int>, std::uint32_t> prev;  // заводи словарь (static перемнная -> один раз инициализируем)

    for (int k = 0; k < num; k ++) {
        float c = lo + k * (hi - lo) / num;  // k-ая изолиния рисуется для такого вот значения константы c
        prev.clear();  // перед вычислением новой изолинии очищаем все предыдущие значения

        for (int i = 1; i <= H; i ++) {
            for (int j = 1; j <= W; j ++) {  // перебираем квадраты (клетки) сетки -> у квадрат (i,j) ровно 4 вершины (= узлы сетки) с координатами (i-1,j-1), (i,j-1), (i,j), (i-1,j)
                /*
                Далее идея проста: разбиваем квадрат на 2 треугольника -> на каждой стороне треугольника ищем точки
                изолинии. Заметим, что точек может быть найдено только чётное количество (0 или 2) (так как в треугольнике три вершины - либо
                все одного знака относительно 'c', а значит изолиния там не проходит (0 точек), либо есть одна точка отличающегося от других знака (изолиния
                проходит через две стороны треугольника, то есть 2 точки)). Эти точки просто добавляем в вектор (точнее их индексы), что означает,
                что отрезочек между этими точками нужно будет нарисовать - он часть линии уровня. Заметим, что мы для каждого такого
                отрезка добаляем подряд две точки (обе его вершины) - просто взяв все пары соседних точек и нарисовав отрезки между ними (что и делает GL_LINES), мы
                нарисуем всю изолинию. Кстати, если изолиний несколько, ничего не меняется - мы просто подряд нарисуем отрезки всех изолиний (а связность каждой
                гарантируется тем, что мы добалвяем точки отрезков в нужно порядке - сначала все точки для первой изолинии, потом другой и тд)
                (это аналог алгоритма marching squares, но для треугольников - так лучше, ведь нет неодноначностей)
                */
                
                int x = 0;  // количество добавленных точек для одного треугольника
                x += add_point_to_contour(j-1, i, j, i, c, points, indices, prev);  // для каждого отрезка (стороны) первого треугольника (половина (i,j)-го квадрата) добавляем точку пересечения с линией уровня
                x += add_point_to_contour(j, i, j-1, i-1, c, points, indices, prev);
                x += add_point_to_contour(j-1, i-1, j-1, i, c, points, indices, prev);

                if (x % 2 != 0) {  // иногда, всё же бывают ситуации, когда добавили нечётное количесто точек (это может быть из-за точности вычислений, а точнее, когда значение в вершине отрезка (числа f1 или f2 из предыд функции) совпали с константой 'c' ->
                                   // чтобы ничего не ломалось (как уже говорили, рисуем мы отрезки для каждой пары точек -> если добавли нечетное количество точек, то при рисовании отрезка его концы будут не внктри одного треугольника, а раскиданы ->
                                   // на картинке будут видны отрезки проведенные как попало...), просто удалим одну из добавленных точек... это может несного подпортить глакость, но если сетка мелкая и так как такая ситуация редка,
                                   // глаз и не заметит)
                    std::cout << "!!! " << c << ":  " << values_function[j-1][i] << " " << values_function[j][i] << " " << values_function[j-1][i-1] << std::endl;  // выведем информацию
                    indices.pop_back();  // удаляем
                }
                x = 0;
                x += add_point_to_contour(j-1, i-1, j, i, c, points, indices, prev);  // аналогично для второго треугольника
                x += add_point_to_contour(j, i, j, i-1, c, points, indices, prev);
                x += add_point_to_contour(j, i-1, j-1, i-1, c, points, indices, prev);
                if (x % 2 != 0) {
                    indices.pop_back();
                    std::cout << "!!! " << c << ":  " << values_function[j-1][i-1] << " " << values_function[j][i] << " " << values_function[j][i-1] << std::endl; 
                }
            }
        }
    }
}



// ====== Код для шейдеров ======
const char vertex_shader_source[] =
R"(#version 330 core

uniform mat4 view;  // из основного кода к нам приходит матрица view, которая должна перевести координаты position из
                    // системы координат сетки ([0, W] на [0, H]) в систему координат OpenGL ([-1, 1] на [-1, 1])

layout (location = 0) in vec2 in_position;  // в вершинный шейдер передаём координаты точки и цвет
layout (location = 1) in vec4 in_color;

out vec4 color;

void main()
{
    gl_Position = view * vec4(in_position, 0.0, 1.0);
    color = in_color;  // цвет далее передаём во фрагментный шейдер
}
)";

const char fragment_shader_source[] =
R"(#version 330 core

in vec4 color;  // из вершинного шейдера приходит цвет (он приходит в каждый пиксель, от котрого вызывается данный фрагментный шейдер, а потому приходит интерполированным для пикселя относительно вершин, в которых его задали)

layout (location = 0) out vec4 out_color;

void main()
{
    out_color = color;  // такого цвета рисуется пиксель
}
)";

GLuint create_shader(GLenum type, const char * source) {
    GLuint result = glCreateShader(type);
    glShaderSource(result, 1, &source, nullptr);
    glCompileShader(result);
    GLint status;
    glGetShaderiv(result, GL_COMPILE_STATUS, &status);
    if (status != GL_TRUE)
    {
        GLint info_log_length;
        glGetShaderiv(result, GL_INFO_LOG_LENGTH, &info_log_length);
        std::string info_log(info_log_length, '\0');
        glGetShaderInfoLog(result, info_log.size(), nullptr, info_log.data());
        throw std::runtime_error("Shader compilation failed: " + info_log);
    }
    return result;
}

GLuint create_program(GLuint vertex_shader, GLuint fragment_shader) {
    GLuint result = glCreateProgram();
    glAttachShader(result, vertex_shader);
    glAttachShader(result, fragment_shader);
    glLinkProgram(result);

    GLint status;
    glGetProgramiv(result, GL_LINK_STATUS, &status);
    if (status != GL_TRUE)
    {
        GLint info_log_length;
        glGetProgramiv(result, GL_INFO_LOG_LENGTH, &info_log_length);
        std::string info_log(info_log_length, '\0');
        glGetProgramInfoLog(result, info_log.size(), nullptr, info_log.data());
        throw std::runtime_error("Program linkage failed: " + info_log);
    }

    return result;
}
// ==================



// Данная функция заполняем буферы координатами сетки и индексами:
uint32_t assign_grid_buffers(GLuint grid_vao, GLuint pos_vbo, GLuint grid_ebo) {
    glBindVertexArray(grid_vao);

    std::vector <position> poses;  // создаём вектор координат вершин сетки и добавляем в него их:
    for (int x = 0; x <= W; x ++) {
        for (int y = 0; y <= H; y ++) {
            position pos = {(float) x, (float) y};
            poses.push_back(pos);
        }
    }
    glBindBuffer(GL_ARRAY_BUFFER, pos_vbo);  // перекопируем данные из этого вектора в буфер (он будет GL_STATIC_DRAW, так как вершины меняться не будут (почти, кроме случаев, когда руками меняем W и H) -> буфер останется неизменным)
    glBufferData(GL_ARRAY_BUFFER, poses.size() * sizeof(position), poses.data(), GL_STATIC_DRAW);

    std::vector <std::uint32_t> indices;  // теперь создаём вектор индексов вершин сетки (чтобы отрисовывать эту сетку индексированным рендерингом)
    for (int i = 1; i <= H; i ++) {
        for (int j = 1; j <= W; j ++) {  // для каждого квадрата (i,j) 
            std::uint32_t v0 = i*(W+1)+j-1;  // перечисляем индексы (индексы внутри poses) его вершин
            std::uint32_t v1 = v0+1;
            std::uint32_t v2 = (i-1)*(W+1)+j-1;
            std::uint32_t v3 = v2+1;
            indices.push_back(v0);  // добавляем (в порядке обхода против часовой) вершины первого треугольника этого квдарата
            indices.push_back(v1);
            indices.push_back(v2);
            indices.push_back(v2);  // теперь для второго треугольник
            indices.push_back(v1);
            indices.push_back(v3);
        }
    }
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, grid_ebo);  // перекопируем в специальный буфер для индексов
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, indices.size() * sizeof(uint32_t), indices.data(), GL_STATIC_DRAW); 

    return indices.size();  // обязательно возвращаем текущее количество индексов
}



int main() try {
    if (SDL_Init(SDL_INIT_VIDEO) != 0)
        sdl2_fail("SDL_Init: ");

    SDL_GL_SetAttribute(SDL_GL_CONTEXT_MAJOR_VERSION, 3);
    SDL_GL_SetAttribute(SDL_GL_CONTEXT_MINOR_VERSION, 3);
    SDL_GL_SetAttribute(SDL_GL_CONTEXT_PROFILE_MASK, SDL_GL_CONTEXT_PROFILE_CORE);
    SDL_GL_SetAttribute(SDL_GL_DOUBLEBUFFER, 1);
    SDL_GL_SetAttribute(SDL_GL_MULTISAMPLEBUFFERS, 1);
    SDL_GL_SetAttribute(SDL_GL_MULTISAMPLESAMPLES, 4);

    SDL_Window * window = SDL_CreateWindow("Graphics course practice 3",
        SDL_WINDOWPOS_CENTERED,
        SDL_WINDOWPOS_CENTERED,
        800, 600,
        SDL_WINDOW_OPENGL | SDL_WINDOW_RESIZABLE | SDL_WINDOW_MAXIMIZED);

    if (!window)
        sdl2_fail("SDL_CreateWindow: ");

    int width, height;
    SDL_GetWindowSize(window, &width, &height);  // получаем размер экрана (точнее, окна, в котором рисуем)

    SDL_GLContext gl_context = SDL_GL_CreateContext(window);
    if (!gl_context)
        sdl2_fail("SDL_GL_CreateContext: ");

    SDL_GL_SetSwapInterval(0);

    if (auto result = glewInit(); result != GLEW_NO_ERROR)
        glew_fail("glewInit: ", result);

    if (!GLEW_VERSION_3_3)
        throw std::runtime_error("OpenGL 3.3 is not supported");

    glClearColor(0.8f, 0.8f, 1.f, 0.f);


   
    // === Настраиваем буферы для рисования сетки (функции цветом по этой сетке):
    GLuint pos_vbo, col_vbo, grid_vao, grid_ebo;
    glGenBuffers(1, &pos_vbo); glGenBuffers(1, &col_vbo);
    glGenVertexArrays(1, &grid_vao); glGenBuffers(1, &grid_ebo);
    
    glBindVertexArray(grid_vao);  // делаем текущим grid_vao буфер (Vertex Array Buffer), внутри него настраиваем аттрибуты вершин:

    glBindBuffer(GL_ARRAY_BUFFER, pos_vbo);  // pos_vbo будет хранить координаты вершин сетки (они не меняются) ->
    glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, sizeof(position), (void*)(0));  // -> из этого буфера берётся 0-ой аттрибут

    glBindBuffer(GL_ARRAY_BUFFER, col_vbo);  // col_vbo будет хранить цвет каждой вершины сетки (вот он меняется какждый кадр -> храним отдельнм буфером)
    glVertexAttribPointer(1, 4, GL_UNSIGNED_BYTE, GL_TRUE, sizeof(color), (void*)(0));  // цвет это 1-ый аттрибут

    glEnableVertexAttribArray(0);  // обязательно включаем аттрибуты (для данного vao) для использования в шейдере 
    glEnableVertexAttribArray(1); 

    uint32_t grid_ind_size = assign_grid_buffers(grid_vao, pos_vbo, grid_ebo);  // заполнили буферы

    

    // === Настройка буфером для рисования линйи уровня:    
    GLuint contour_vbo, contour_vao, contour_ebo, black_color_vbo;
    glGenBuffers(1, &contour_vbo); glGenBuffers(1, &contour_ebo);
    glGenVertexArrays(1, &contour_vao); glGenBuffers(1, &black_color_vbo);  // создаём (не забываем, что буфер vao создаётся командой glGenVertexArrays, а не glGenBuffers - долго ошибка была...)

    glBindVertexArray(contour_vao);  // делаем текущим и снова настраиваем аттрибуты, но уже для contour_vao
    
    glBindBuffer(GL_ARRAY_BUFFER, contour_vbo);  // позиции вершин (точек концов отрезкоа линий уровня) хранятся в данном бкфере
    glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, sizeof(position), (void*)(0));

    glBindBuffer(GL_ARRAY_BUFFER, black_color_vbo);  // цвет всех линий уровня будет чёрным, поэтому сразу заполяем его и тоже аттрибут настроим
    uint8_t black[4] = {0, 0, 0, 255};  // цвет = 4 раза по 8 бит (в данном случае цветовые каналы 0 (черный цвет), а канал прозрачности = 255 (непрозрачный))
    glBufferData(GL_ARRAY_BUFFER, 4 * sizeof(uint8_t), black, GL_STATIC_DRAW);  // копируем в буфер (он GL_STATIC_DRAW, так как не меняется... вообще интересно: для сетки у нас координаты постоянны, а меняется цвет, а для изолиний противоположно)
    glVertexAttribPointer(1, 4, GL_UNSIGNED_BYTE, GL_TRUE, 0, (void*)(0));
    
    glEnableVertexAttribArray(0);  // также включаем аттрибуты
    glEnableVertexAttribArray(1); 
    


    // === Основное рисование:
    auto vertex_shader = create_shader(GL_VERTEX_SHADER, vertex_shader_source);
    auto fragment_shader = create_shader(GL_FRAGMENT_SHADER, fragment_shader_source);
    auto program = create_program(vertex_shader, fragment_shader);  // получили шейдерную программу

    GLuint view_location = glGetUniformLocation(program, "view");  // получили указатель на переменную в этой прогамме
    auto last_frame_start = std::chrono::high_resolution_clock::now();
    float time = 0.f;
    bool running = true;
    int num_contours = 3;  // количество рисуемых изолиний
    while (running) {
        for (SDL_Event event; SDL_PollEvent(&event);) switch (event.type)
        {
        case SDL_QUIT:
            running = false;
            break;
        case SDL_WINDOWEVENT: switch (event.window.event)
            {
            case SDL_WINDOWEVENT_RESIZED:
                width = event.window.data1;
                height = event.window.data2;
                glViewport(0, 0, width, height);
                break;
            }
            break;
        case SDL_KEYDOWN:  // обрабатываем нажатия клавиш!
            if (event.key.keysym.sym == SDLK_LEFT && num_contours > 1) {
                num_contours -= 1;  // клавиши ВЛЕВО/ВПРАВО меняет количество изолиний
            }
            else if (event.key.keysym.sym == SDLK_RIGHT) {
                num_contours += 1;
            }
            else if (event.key.keysym.sym == SDLK_DOWN && H >= 50 && W >= 50) {
                H -= 25;
                W -= 25;  // клавиши ВВЕРХ/ВНИЗ изменяют детализацию сетки
                grid_ind_size = assign_grid_buffers(grid_vao, pos_vbo, grid_ebo);
            }
            else if (event.key.keysym.sym == SDLK_UP  && W+25 <= 1000 && H+25 <= 1000) {  // ограничения <= 1000, так как массив function_values таков
                H += 25;
                W += 25;
                grid_ind_size = assign_grid_buffers(grid_vao, pos_vbo, grid_ebo);  // изменяя W и H приходится поменять и буферы, отвечающие за вершины сетки
            }
            break;
        }

        if (!running)
            break;

        sample_on_grid(time);  // в начале кадра (но после того как W и H могли поменяться) сразу вычисляем значения функции


        // вычисляем цвет вершин сетки и записываем их в буфер (буфер GL_DYNAMIC_DRAW, так как меняется постоянно)
        std::vector <color> cols = get_color();
        glBindVertexArray(grid_vao);
        glBindBuffer(GL_ARRAY_BUFFER, col_vbo); 
        glBufferData(GL_ARRAY_BUFFER, cols.size() * sizeof(color), cols.data(), GL_DYNAMIC_DRAW);                    


        // вычисляем изолинии и тоже в буферы кладём
        std::vector <position> points;
        std::vector <uint32_t> contour_ind;
        get_contour(points, contour_ind, num_contours);
        glBindVertexArray(contour_vao);
        glBindBuffer(GL_ARRAY_BUFFER, contour_vbo); 
        glBufferData(GL_ARRAY_BUFFER, points.size() * sizeof(position), points.data(), GL_DYNAMIC_DRAW);    
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, contour_ebo); 
        glBufferData(GL_ELEMENT_ARRAY_BUFFER, contour_ind.size() * sizeof(uint32_t), contour_ind.data(), GL_DYNAMIC_DRAW);    
        

        // считаем время, которое прошло
        auto now = std::chrono::high_resolution_clock::now();
        float dt = std::chrono::duration_cast<std::chrono::duration<float>>(now - last_frame_start).count();
        last_frame_start = now;
        time += dt;
        glClear(GL_COLOR_BUFFER_BIT);  // очищаем цветовой буфер

        float view[16] =  // матрица, которая переводит координаты из сетки в экран
        {
            1.6f/W, 0.f, 0.f, -0.8f,   // очевидно, что если x \in [0, W], то после применения к (x, y, 0, 1) первая координата будет от -0.8 до 0.8 (то есть нарисованное окошко сетки будет занимать 2 * 0.8 / 2 = 80% экрана по ширине)
            0.f, -1.6f/H, 0.f, +0.8f,  // похожее с y
            0.f, 0.f, 1.f, 0.f,
            0.f, 0.f, 0.f, 1.f,
        };

        float aspect_ratio = width * 1.f / height;  // считаем aspect ratio окна, которое открыто в данный момент
        // подправляем матрицу, чтобы при изменении соотношения сторон окна, сетка оставалась с исходным соотношением сторон,
        // а также, чтобы сетка целиком влезла в окно:
        if (aspect_ratio > 1) {  // если ширина окна > высоты
            view[0] /= aspect_ratio;  
            view[3] /= aspect_ratio;  // уменьшаем x-координату (чтобы y тоже была меньше и влезла по высоте)
        } else {
            view[5] *= aspect_ratio;  // наоборот - уменьшаем y
            view[7] *= aspect_ratio;
        }

        // устанавливаем матрицу:
        glUseProgram(program);
        glUniformMatrix4fv(view_location, 1, GL_TRUE, view);
        

        // отрисовываем сетку:
        glBindVertexArray(grid_vao);
        glDrawElements(GL_TRIANGLES, grid_ind_size, GL_UNSIGNED_INT, (void*)(0));

        // отрисовываем линии уровня:
        glBindVertexArray(contour_vao); 
        glDrawElements(GL_LINES, contour_ind.size(), GL_UNSIGNED_INT, (void*)(0));
        glLineWidth(2.f);  // делаем лини потолще


        // отправляем команду отрисовки 
        SDL_GL_SwapWindow(window);
    }


    SDL_GL_DeleteContext(gl_context);
    SDL_DestroyWindow(window);
}
catch (std::exception const & e) {
    std::cerr << e.what() << std::endl;
    return EXIT_FAILURE;
}
