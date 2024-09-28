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

std::string to_string(std::string_view str)
{
    return std::string(str.begin(), str.end());
}

void sdl2_fail(std::string_view message)
{
    throw std::runtime_error(to_string(message) + SDL_GetError());
}

void glew_fail(std::string_view message, GLenum error)
{
    throw std::runtime_error(to_string(message) + reinterpret_cast<const char *>(glewGetErrorString(error)));
}


/* было изначально, достаточно для заданий 1-7, для 8 задания нужно немного подправить шейдеры...
const char vertex_shader_source[] =
R"(#version 330 core

uniform mat4 view;

layout (location = 0) in vec2 in_position;
layout (location = 1) in vec4 in_color;

out vec4 color;

void main()
{
    gl_Position = view * vec4(in_position, 0.0, 1.0);
    color = in_color;
}
)";

const char fragment_shader_source[] =
R"(#version 330 core

in vec4 color;

layout (location = 0) out vec4 out_color;

void main()
{
    out_color = color;
}
)";
*/


// === Задание 8 ===
const char vertex_shader_source[] =
R"(#version 330 core

uniform mat4 view;

layout (location = 0) in vec2 in_position;
layout (location = 1) in vec4 in_color;
layout (location = 2) in float in_dist;  // передаём третий аттрибут - расстояние до вершины от начала на кривой Безье (этот аттрибут
                                        // будет передаваться только для вершин на кривой Безье, для ломанной он будет видимо заменён дефолтным знаением... но это не важно, так как для ломанной использоваться не будет)
out float dist;  // этот выходной параметр, с помощью которого in_dist просто передадим в пиксельный шейдер (важно, что пиксельный шейдер вызвватся не для вершин, а для каждого пикселя -> значене этого параметра будет интерполировано 
                 // как и цвет обычно... интерполяция для ломанной происходит для каждого отрезка через две его вершины)
out vec4 color;

void main()
{
    gl_Position = view * vec4(in_position, 0.0, 1.0);
    color = in_color;
    dist = in_dist;
}
)";

const char fragment_shader_source[] =
R"(#version 330 core

in vec4 color;
in float dist;  // из вершинного шейдера в пиксельный (после интерполяции) приходит расстояние от начала кривой Безье (только для пикселей кривой Безье, для пикселей ломанной неважно что тут)
uniform int dash;  // переменная, которую ставим извне - она 1 (если сейчас рисуем кривую Безье и хотим рисовать её пунктиром)
uniform float time;  // также передаём текущее время - оно понадобится, чтобы пунктир двигался (опять же, это нужно исключительно для рисования криой Безье)

layout (location = 0) out vec4 out_color;

void main()
{
    if (dash == 1 && mod(dist+20*time, 40.0) < 20.0)  // если используем пунктир (dash = 1) и при этом остаток от деления на 40 меньше 20, то пиксель не рисуем - просто discard вызываем
        discard;                                      // (к расстоянию с некоторым весом добавляем время, что пунктир каждый раз рисовался по-разному - кажется, что двигается)
    out_color = color;
}
)";


GLuint create_shader(GLenum type, const char * source)
{
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

GLuint create_program(GLuint vertex_shader, GLuint fragment_shader)
{
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

struct vec2
{
    float x;
    float y;
};

struct vertex
{
    vec2 position;
    std::uint8_t color[4];
};

vec2 bezier(std::vector<vertex> const & vertices, float t)
{
    std::vector<vec2> points(vertices.size());

    for (std::size_t i = 0; i < vertices.size(); ++i)
        points[i] = vertices[i].position;

    // De Casteljau's algorithm
    for (std::size_t k = 0; k + 1 < vertices.size(); ++k) {
        for (std::size_t i = 0; i + k + 1 < vertices.size(); ++i) {
            points[i].x = points[i].x * (1.f - t) + points[i + 1].x * t;
            points[i].y = points[i].y * (1.f - t) + points[i + 1].y * t;
        }
    }
    return points[0];
}

int main() try
{
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
    SDL_GetWindowSize(window, &width, &height);  // получаем размер экрана

    SDL_GLContext gl_context = SDL_GL_CreateContext(window);
    if (!gl_context)
        sdl2_fail("SDL_GL_CreateContext: ");

    SDL_GL_SetSwapInterval(0);

    if (auto result = glewInit(); result != GLEW_NO_ERROR)
        glew_fail("glewInit: ", result);

    if (!GLEW_VERSION_3_3)
        throw std::runtime_error("OpenGL 3.3 is not supported");

    glClearColor(0.8f, 0.8f, 1.f, 0.f);


    /*  // до задания 4 было так:
    // === Задание 1: ===
    vertex vertices[3] = {{{120, 300}, {255, 0, 0, 255}},
                          {{800, 400}, {0, 255, 0, 255}},
                          {{550, 800}, {0, 0, 255, 255}}};  // создаём и заполняем массив для 3 вершин типа vertex
                                                            // (координаты задаём в пикселях - как в задании 3 сказано; цвет задаём как
                                                            // число от 0 (значит данный цветовой канал не светится) до 255 (макс яркость) -> далее это число отнормируется при загрузке аттрибута в вершинный шейдер и станет от 0 до 1)

    GLuint vbo;
    glGenBuffers(1, &vbo);  // создаём один объект типа буфер (это буфер в памяти именно видеокарты, не обычной оперативке), его едентификатор помещается в переменную vbo
    glBindBuffer(GL_ARRAY_BUFFER, vbo);  // делаем данный буфер (с меткой (target) GL_ARRAY_BUFFER) текущим ->
                                         // -> это означает, что все последующие функции, в которых target=GL_ARRAY_BUFFER будут работать именно с этим буфером (vbo)
    glBufferData(GL_ARRAY_BUFFER, 3 * sizeof(vertex), vertices, GL_STATIC_DRAW);  // загружаем в созданный буфер данные - массив из 3-ёх вершин (как и говорилось, указываем GL_ARRAY_BUFFER -> автоматически работаем с vbo, так как именно он текущий)
                                                                                  // указываем GL_STATIC_DRAW, так как использовать будем только для рисования (DRAW, на чтение) и не будем менять (STATIC)
    
    vertex temp;  // создаём временную переменную для вершины
    glGetBufferSubData(GL_ARRAY_BUFFER, 0, sizeof(vertex), &temp);  // считываем в неё первую загруженную вершину
    std::cout << temp.position.x << " " << temp.position.y << " " << (int) temp.color[0] << std::endl;  // выводим её позицию и первый цвет (просто, чтобы проверить, что все ок записалось в буфер);
                                                                                                        // важно! temp.color[0] имеет тип uint8_t, который является typedef на char (= 1 байт) -> а тип char терминал воспринимает как символ (он впринципе любой единичный байт, который к нему приходит воспринимает в качестве ascii-кода символа) ->
                                                                                                        // -> вместо числа пытается вывести символ с таким номером, которого скорее всего нет - будет непечатный символ... -> для получения числа нужно скастить к int, например (в этом случае на терминал придёт уже не однакий байт, а сразу 4 (так как размер int = 4) ->
                                                                                                        // -> такое терминал уже как просто число воспринимает)
    // === Задание 2: ===
    GLuint vao;
    glGenVertexArrays(1, &vao);  // создаём объект типа Vertex Array
    glBindVertexArray(vao);  // делаем его текущим - эта важно, так как далее мы настриваем аттрибуты вершин, что запоминается в vao
                             // (тут уже никаких target нет -> абсолютно все дальнейшие функции, использующие vao, будут работать с установленным тут "текущим vao" ->
                             // -> с одной стороны удобно (не нужно как параметр везде в функции передавать), с другой - слшком неявно... для изменения vao придётся делать текущим другой vao)
    
    // согласно коду вершинного шейдера, он ждёт два аттрибута: с индексом 0 (это координаты вершины), с индекс 1 (это цвет вершины) -> настраиваем:
    glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, sizeof(vertex), (void*)(0));  // настраиваем аттрибуты вершин vertex как на лекции - помним, что сами вершины берутся из того vbo, который был bind (сделан текущим) последним 
    glVertexAttribPointer(1, 4, GL_UNSIGNED_BYTE, GL_TRUE, sizeof(vertex), (void*)(sizeof(vec2)));  // (а сами аттрибуты запоминаются в vao!)
    glEnableVertexAttribArray(0);  // включаем аттрибуты по индексу, чтобы использовать в вершинном шейдере
    glEnableVertexAttribArray(1);
    */


    // === Задание 4: ===
    std::vector <vertex> vertices;  // создали вектор в/из которого будут добавляться/удаляться вершины по клику мышкой
    GLuint vbo, vao;
    glGenBuffers(1, &vbo); glBindBuffer(GL_ARRAY_BUFFER, vbo);  // создали vbo и сделали его текущим (пока данные в него не копируем, так как вершин в векторе нет) 
    glGenVertexArrays(1, &vao); glBindVertexArray(vao);  // аналогично создали с делали текущим vao с помощью bind

    // как и ранее, настраиваем аттрибуты вершин - заметим, что это можно сделать прямо тут, так как от изменения числа вершин
    // в будущем ничего не поменяется в аттрибутах: нам лишь сейчас нужно настроить аттрибуты, указав индекс и размер каждого из них,
    // а также указав смещение и stride, по которому они лежат (а точнее в нашем случае - будут лежать) в vbo (именно в том vbo, который сейчас текущий) (очевидно, что
    // если добавлять вершины единообразно, то ни stride, ни смещение, ни тем более индекс и размер аттрибутов не поменяются -> достаточно
    // настроить только тут, не придётся настраивать аттрибуты в цикле);
    // сразу заметим, что так как в vbo мы будем хранить просто экземпляры vertex друг за другом, то смещение stride = sizeof(vertex) для всех аттрибутов!
    glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, sizeof(vertex), (void*)(0));  // аттрбут с индексом 0 состоит из 2 компонент (рамер = 2) типа float (он отвечает за координат x,y вершины); так как в vbo мы будем хранить просто экземпляры vertex друг за другом, а координаты - первое поле в vertex, то смещение (для данного аттрибута относительно начала vbo) равно 0  
    glVertexAttribPointer(1, 4, GL_UNSIGNED_BYTE, GL_TRUE, sizeof(vertex), (void*)(sizeof(vec2)));  // аттрибут с индексом 1 состоит из 4 компоент типа uint8_t, причём мы их нормируем (GL_TRUE указываем) (чтобы в вершинном шейдере оказывались числа от 0 до 1) - этот аттрибут это 4 компоненты цвета; так как поле цвета лежит в vertex сразу за кординатами, то смещение = рамзер коорднат =  sizeof(vec2)
    glEnableVertexAttribArray(0);
    glEnableVertexAttribArray(1);  // снова включаем аттрибуты
    // заметим, что мы нигде при настройке аттрбутов не указываем количество вершин - это логично (и особенно удобно в данном задании, где количество вершин переменное), так как
    // количество вершинок указывается в glDrawArrays (уже внутри цикла для конкретной итерации рисования), а далее просто для каждого индекса вершины (от 0 до их количества) высчитывается (с помощью
    // указанного нами при настройке аттрибутов смещения и stride) расположение соответствующего аттрибута -> он достаётся из памяти (из соответствующего vbo, который был bind при настройке аттрибутов - эта информация (какой vbo при настройке данного аттрибута был)
    // вместе с самими настройками вроде смещения и stride запомнены внутри vao, который должен быть bind при вызове glDrawArrays) (если его там нет, то будет видимо segmntation fault) и передаётся в вершинный шейдер.


    // === Задание 6: ===
    // заводим отдельные объекты для хранения и рисования кривой Безье - они полностью аналогичны предыдущим
    std::vector <vertex> bezier_vertices;
    GLuint bezier_vbo, bezier_vao;
    glGenBuffers(1, &bezier_vbo); glBindBuffer(GL_ARRAY_BUFFER, bezier_vbo);
    glGenVertexArrays(1, &bezier_vao); glBindVertexArray(bezier_vao);
    glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, sizeof(vertex), (void*)(0));
    glVertexAttribPointer(1, 4, GL_UNSIGNED_BYTE, GL_TRUE, sizeof(vertex), (void*)(sizeof(vec2))); 
    glEnableVertexAttribArray(0);
    glEnableVertexAttribArray(1);
    int quality = 4;  // добавляем переменную детализации кривой



    auto vertex_shader = create_shader(GL_VERTEX_SHADER, vertex_shader_source);
    auto fragment_shader = create_shader(GL_FRAGMENT_SHADER, fragment_shader_source);
    auto program = create_program(vertex_shader, fragment_shader);


    // === Задание 8: ===
    std::vector <float> dists_bezier;  // создаём отдельный вектор, где для каждой вершины кривой Безье будет находиться расстояние вдоль кривой от её начала до этой вершины
    GLuint dists_vbo; glGenBuffers(1, &dists_vbo); glBindBuffer(GL_ARRAY_BUFFER, dists_vbo);  // для этого же заводим свой vbo, делаем его текущим
    glVertexAttribPointer(2, 1, GL_FLOAT, GL_FALSE, 0, (void*)(0));  // и после (так как аттрибут будет лежать в dists_vbo, нужно сначала его сделать текущим) этого регистрируем новый аттрибут (число float, 1 компонента), которая будет как раз этим расстоянием
    // (заметим, что в качестве stride передаём 0, так как dists_vbo будет хранить подряд только эти расстояния -> stride сам посчитается как sizeof(float))
    // (также заметим, что мы не заводим новый vao, новый аттрибут будет запомнен в bezier_vao, который и используется для рисования кривой Безье)
    glEnableVertexAttribArray(2);  // обязательно включает аттрибут, иначе он не будет передаваться в шейдер!!!
    GLuint dash_location = glGetUniformLocation(program, "dash");  // находим положение переменной для обозначения необходимости рисования Безье пунктиром
    GLuint time_location = glGetUniformLocation(program, "time");



    GLuint view_location = glGetUniformLocation(program, "view");
    auto last_frame_start = std::chrono::high_resolution_clock::now();
    float time = 0.f;
    bool running = true;
    while (running)
    {
        // === Задание 6 ===
        bool recompute_bezier = false;  // переменная показывает, нужно ли на данной итерации пересчитывать кривую Безье


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

        case SDL_MOUSEBUTTONDOWN:  // обрабатываем клик мыши !!!
            if (event.button.button == SDL_BUTTON_LEFT)
            {
                int mouse_x = event.button.x;  // запоминаем координаты мыши при клике
                int mouse_y = event.button.y;  // (заметим, что коорднаты мыши - это целые числа в координатах в пискелях экрана - то есть как мы и хотим задавать вершиы, начиная с задания 3)

                // === Задание 4 ===
                vertex new_v = {(float) mouse_x, (float) mouse_y, 59, 68, 75, 255};  // создали вершину какого-то цвета с координатами мыши
                vertices.push_back(new_v);  // добавили в вектор эту вершину -> теперь ломанная получила + 1 отрезок с концом в жанной точке
                glBindBuffer(GL_ARRAY_BUFFER, vbo);  // делаем vbo текущим (в принципе он им уже и так был, но на всякий случай - если в будущем будет ольше кода с другими буферами, которые тоже будут в некоторые моменты текущими)
                glBufferData(GL_ARRAY_BUFFER, vertices.size() * sizeof(vertex), vertices.data(), GL_STATIC_DRAW);  // перезаписываем vbo данными вершин из вектора (в этот момент старые данные, если они были, выбрасываются, а память переаллоцируется... часто кликать мышкой не стоит)
            
                // === Задание 6 ===
                recompute_bezier = true;  // так как ломанная изменилась (добавлась точка), придётся пересчитывать Безье
            }
            else if (event.button.button == SDL_BUTTON_RIGHT)
            {
                // === Задание 4 ===
                if (vertices.size() > 0) {  // если вектор не пустой, выкидываем последнюю вершину и перезаписываем vbo
                    vertices.pop_back();
                    glBindBuffer(GL_ARRAY_BUFFER, vbo); 
                    glBufferData(GL_ARRAY_BUFFER, vertices.size() * sizeof(vertex), vertices.data(), GL_STATIC_DRAW); 
                    recompute_bezier = true;  // аналогично
                }
            }
            break;

        case SDL_KEYDOWN:  // обрабатываем нажатия клавиш!
            if (event.key.keysym.sym == SDLK_LEFT)
            {   
                // === Задание 7 ===
                if (quality > 1) {  // при нажатии клавиши "стрелочка влево" уменьшаем детализацию (но не меньше 1)
                    quality -= 1;
                    recompute_bezier = true;  // не забываем поставить флаг, что нужно пересчитать кривую Безье
                }
            }
            else if (event.key.keysym.sym == SDLK_RIGHT)
            {
                // === Задание 7 ===
                quality += 1;  // при нажатии "стрелочка вправо" увеличиваем детализацию
                recompute_bezier = true;
            }
            break;
        }

        if (!running)
            break;


        // === Задание 6 ===
        if (recompute_bezier) {  // если нужно, пересчитываем кривую Безье
            bezier_vertices.clear();  // очищаем старые точки кривой Безье, сейчас будем считать заново
            int number_bezier_segments = quality * (vertices.size() - 1);  // количество отрезков, которыми приближается кривая Безье (чтобы количество отрезков было ровно в quality раз больше, чем на ломанной лини из vertices); для этого заметим, что если n - число точек, то в ломанной между ними n-1 отрезок
            for (int i = 0; i <= number_bezier_segments; i ++) {
                vec2 p = bezier(vertices, float(i) / number_bezier_segments);  // равномерно (с шагом dt = 1 / (number_bezier_segments)) (по парамтреу t от 0 до 1) высчитываем координаты точек на кривой Безье (а между этими точками кривая будет приближаться просто отрезком -> вся кривая будет приближаться ломанной... при достаточно больших quality приюлижение будет хорошим)
                vertex new_v = {p.x, p.y, 255, 0, 0, 255};  // добавляем к точке цвет - получаем вершину кривой
                bezier_vertices.push_back(new_v);  // добавили в вектор
            }
            glBindBuffer(GL_ARRAY_BUFFER, bezier_vbo);  // теперь делаем bezier_vbo текущим, так как с ним сейчас будем работать 
            glBufferData(GL_ARRAY_BUFFER, bezier_vertices.size() * sizeof(vertex), bezier_vertices.data(), GL_STATIC_DRAW);  // копируем в него полученные точки кривой Безье
        
            // === Задание 8 ===
            dists_bezier.assign(bezier_vertices.size(), 0.0);  // создаём массив расстояний вершин от начала, пока из 0
            for (int i = 1; i < dists_bezier.size(); i ++) {
                float dx = bezier_vertices[i].position.x - bezier_vertices[i-1].position.x;  // смотрим смещение по координатам для отрезка, приближающего кривую, между i-1 и i вершинами кривой Безье
                float dy = bezier_vertices[i].position.y - bezier_vertices[i-1].position.y;
                dists_bezier[i] = dists_bezier[i-1] + sqrt(dx * dx + dy * dy);  // вычисляем расстояние до i-ой вершины кривой Безье от её начала динамическим программированием
                                                                                // (расстояние до i-ой точки = расстояние до i-1-ой + длина отрезка от i-1 до i; длина отрезка просто по теореме Пифагора...)
            }   

            glBindBuffer(GL_ARRAY_BUFFER, dists_vbo);  // полученный вектор расстояний используем, запллняя нужный vbo
            glBufferData(GL_ARRAY_BUFFER, dists_bezier.size() * sizeof(float), dists_bezier.data(), GL_STATIC_DRAW);                    
        }


        auto now = std::chrono::high_resolution_clock::now();
        float dt = std::chrono::duration_cast<std::chrono::duration<float>>(now - last_frame_start).count();
        last_frame_start = now;
        time += dt;

        glClear(GL_COLOR_BUFFER_BIT);

        /* было изначально
        float view[16] =
        {
            1.f, 0.f, 0.f, 0.f,
            0.f, 1.f, 0.f, 0.f,
            0.f, 0.f, 1.f, 0.f,
            0.f, 0.f, 0.f, 1.f,
        };
        */

        // === Задание 3 ===
        // здесь мы задаём 4на4 матрицу, которая применится (в вершинном шейдере) к вектору вида (x, y, 0, 1) - последни две координаты это
        // число 0 (так как третьей координаты z пока нет) и число 1 (нужно в проектвных координатах, чтобы показать, что это именно точка, а не вектор);
        // мы хотим задавать x и y как координаты на экране в пикселях: x от 0 до width (это слева направо) и y от height до 0 (сверху вниз) ->
        // но OpenGL хочет координаты от -1 до 1 по обеим -> данная матрица должна делать такое превращение:
        float view[16] =
        {
            2.f/width, 0.f, 0.f, -1.f,   // очевидно, что если x \in [0, width], то после применения к (x, y, 0, 1) первая координата будет от -1 до 1 (как и хотели)
            0.f, -2.f/height, 0.f, +1.f,  // похожее с y
            0.f, 0.f, 1.f, 0.f,
            0.f, 0.f, 0.f, 1.f,
        };


        glUseProgram(program);
        glUniformMatrix4fv(view_location, 1, GL_TRUE, view);
        
        // === Задание 8 ===
        glUniform1i(dash_location, 0);  // пока ставим dash = 0, потому при рисовании ломанной никакой пунктир не нужен

        /*
        // === Задание 2 ===
        glDrawArrays(GL_TRIANGLES, 0, 3);  // рисуем треугольник: с 0-ой вершины берётся 3 штуки - формально просто аттрибуты берутся из запомненных в vao буферов 
        */

        // === Задание 4 ===
        glBindVertexArray(vao);  // на всякий случай снова делаем vao текущим (хотя он и так им был, но вдруг потом еще vao насоздаём и будем их делать текущими)
        glDrawArrays(GL_LINE_STRIP, 0, vertices.size());  // рисуем все vertices.size() вершин, начная с 0-ой (все эти вершины, а точнее их аттрбуты, уже должны быть в vbo (который запомнен внутр vao, который сделали bind)) - оттуда они в этой функции будут вытаскиваться и передаваться в вершинный шейдер)
        glLineWidth(5.f);  // делаем лини потолще

        // === Задание 5 ===
        glDrawArrays(GL_POINTS, 0, vertices.size());  // дополнительно рисуем толстенькие точки, чтобы заметны были
        glPointSize(10);


        // === Задание 8 ===
        glUniform1i(dash_location, 1);  // теперь перед рисованием кривой Безье включаем пунктир
        glUniform1f(time_location, time);  // также нужно время

        // === Задание 6 ===
        glBindVertexArray(bezier_vao);  // обязательно делаем bind нового vao, так как сейчас уже работаем с vao, отвечающм за кривую Безье
        glDrawArrays(GL_LINE_STRIP, 0, bezier_vertices.size());  // аналогично рисуем ломанную, но уже по точкам Безье
        glLineWidth(5.f);

        SDL_GL_SwapWindow(window);
    }

    SDL_GL_DeleteContext(gl_context);
    SDL_DestroyWindow(window);
}
catch (std::exception const & e)
{
    std::cerr << e.what() << std::endl;
    return EXIT_FAILURE;
}
