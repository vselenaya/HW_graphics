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

#include "obj_parser.hpp"

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

const char vertex_shader_source[] =
R"(#version 330 core

uniform mat4 model;
uniform mat4 view;
uniform mat4 projection;

layout (location = 0) in vec3 in_position;
layout (location = 1) in vec3 in_normal;

out vec3 normal;

void main()
{
    gl_Position = projection * view * model * vec4(in_position, 1.0);
    normal = normalize(mat3(model) * in_normal);
}
)";

const char fragment_shader_source[] =
R"(#version 330 core

in vec3 normal;

layout (location = 0) out vec4 out_color;

void main()
{
    vec3 ambient_dir = vec3(0.0, 1.0, 0.0);
    vec3 ambient_color = vec3(0.2);

    vec3 light1_dir = normalize(vec3( 3.0, 2.0,  1.0));
    vec3 light2_dir = normalize(vec3(-3.0, 2.0, -1.0));

    vec3 light1_color = vec3(1.0,  0.5, 0.25);
    vec3 light2_color = vec3(0.25, 0.5, 1.0 );

    vec3 n = normalize(normal);

    vec3 color = (0.5 + 0.5 * dot(n, ambient_dir)) * ambient_color
        + max(0.0, dot(n, light1_dir)) * light1_color
        + max(0.0, dot(n, light2_dir)) * light2_color
        ;

    float gamma = 1.0 / 2.2;
    out_color = vec4(pow(min(vec3(1.0), color), vec3(gamma)), 1.0);
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
    SDL_GL_SetAttribute(SDL_GL_RED_SIZE, 8);
    SDL_GL_SetAttribute(SDL_GL_GREEN_SIZE, 8);
    SDL_GL_SetAttribute(SDL_GL_BLUE_SIZE, 8);
    SDL_GL_SetAttribute(SDL_GL_DEPTH_SIZE, 24);

    SDL_Window * window = SDL_CreateWindow("Graphics course practice 4",
        SDL_WINDOWPOS_CENTERED,
        SDL_WINDOWPOS_CENTERED,
        800, 600,
        SDL_WINDOW_OPENGL | SDL_WINDOW_RESIZABLE | SDL_WINDOW_MAXIMIZED);

    if (!window)
        sdl2_fail("SDL_CreateWindow: ");

    int width, height;
    SDL_GetWindowSize(window, &width, &height);

    SDL_GLContext gl_context = SDL_GL_CreateContext(window);
    if (!gl_context)
        sdl2_fail("SDL_GL_CreateContext: ");

    if (auto result = glewInit(); result != GLEW_NO_ERROR)
        glew_fail("glewInit: ", result);

    if (!GLEW_VERSION_3_3)
        throw std::runtime_error("OpenGL 3.3 is not supported");

    // строки, по которым можно узнать, какая видеокарта используется (её модель -> можно отличить
    // интегрированную в процессор intel от дискретной nvidea)
    std::cout << glGetString(GL_VENDOR) << std::endl;
    std::cout << glGetString(GL_RENDERER) << std::endl;

    glClearColor(0.1f, 0.1f, 0.2f, 0.f);

    auto vertex_shader = create_shader(GL_VERTEX_SHADER, vertex_shader_source);
    auto fragment_shader = create_shader(GL_FRAGMENT_SHADER, fragment_shader_source);
    auto program = create_program(vertex_shader, fragment_shader);

    GLuint model_location = glGetUniformLocation(program, "model");
    GLuint view_location = glGetUniformLocation(program, "view");
    GLuint projection_location = glGetUniformLocation(program, "projection");

    std::string project_root = PROJECT_ROOT;
    obj_data bunny = parse_obj(project_root + "/bunny.obj");  // загружаем из файла модель зайца - его вершины и индексы вершин

    auto last_frame_start = std::chrono::high_resolution_clock::now();

    float time = 0.f;

    std::map<SDL_Keycode, bool> button_down;



    // =========  МОЙ КОД ================


    // === Задание 1 ===
    GLuint vbo, vao, ebo;

    glGenBuffers(1, &vbo);  // создали буфер 
    glBindBuffer(GL_ARRAY_BUFFER, vbo);  // сделали его текущим буфером среди буфером с target = GL_ARRAY_BUFFER
    glBufferData(GL_ARRAY_BUFFER, bunny.vertices.size() * sizeof(obj_data::vertex), bunny.vertices.data(), GL_STATIC_DRAW);  // загрузи в текущий буфер GL_ARRAY_BUFFER (то есть как раз в vbo) вершины зайца

    glGenVertexArrays(1, &vao);  // создали буфер для vao
    glBindVertexArray(vao);  // сделали его текущим
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, sizeof(obj_data::vertex), (void*)(0));  // настроили аттрибуты вершин, учитывая, что они лежат в текущем GL_ARRY_BUFFER, то есть в vbo
    glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, sizeof(obj_data::vertex), (void*)(sizeof(obj_data::vertex::position))); 
    glEnableVertexAttribArray(0);  // активировали аттрибуты (с индексом 0 и 1)
    glEnableVertexAttribArray(1);

    glGenBuffers(1, &ebo);  // создали буфер под индексы
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, ebo); 
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, bunny.indices.size() * sizeof(uint32_t), bunny.indices.data(), GL_STATIC_DRAW);  // загрузили индексы


    // === Задание 4 ===
    glEnable(GL_DEPTH_TEST); // включаем тест глубины

    // === Задание 5 ===
    float bunny_x = 0, bunny_y = 0;  // задаём изначальные координаты (а точнее смещение по осям) зайца (его центра)
                                     // (причём это в системе координат, в которой заяц задан -> изначально пишем 0,0, имея в виду, что смещения нет - то есть как был записан заяц в файлике в его координатах, такой и есть)

    bool running = true;
    while (running)
    {
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
        case SDL_KEYDOWN:
            button_down[event.key.keysym.sym] = true;
            break;
        case SDL_KEYUP:
            button_down[event.key.keysym.sym] = false;
            break;
        }

        if (!running)
            break;

        auto now = std::chrono::high_resolution_clock::now();
        float dt = std::chrono::duration_cast<std::chrono::duration<float>>(now - last_frame_start).count();
        last_frame_start = now;
        time += dt;

        glClear(GL_COLOR_BUFFER_BIT);

        // === Задание 4 ===
        glClear(GL_DEPTH_BUFFER_BIT);  // очищаем буфер глубины в начале каждого кадра (чтобы там не остались значения с предыдущего
                                       // кадра, которые испортят отрисовку - аналогично, как очищаем буфер с цветом в предыдущей строке)
        // === Задание 6 ===
        glEnable(GL_CULL_FACE);  // включаем back-face culling (будут отрисовываться треугольники, заданные против часовой стрелке, а остальные - нет)
                                 // (конкретно это ничего не меняет, так как заяц задан так, чтобы при просмотре извне все треугольники были против часовой)
        //glCullFace(GL_FRONT);  // а вот если поменять (вместо треугольников против часовой рисовать те, что по часовой), то будет странно выглядеть:
                               // смотря на зайца спереди, мы будем видеть внутреннюю стенку его задней части (как если бы настоящий заяц был наполовину прозрачный)                        
        
        /*
        float model[16] =
        {
            1.f, 0.f, 0.f, 0.f,
            0.f, 1.f, 0.f, 0.f,
            0.f, 0.f, 1.f, 0.f,
            0.f, 0.f, 0.f, 1.f,
        };
        */

        /*
        // === Задание 2 ===
        float angle = time;  // в качестве угла поворота задаём время -> время менятся - угол тоже -> заяц вращается
        float scale = 0.5;  // масштаб
        float model[16] =
        {                                                               // эта матрица является композицией (те произведением) матрицы изменения масштаба A и матрицы поворота-смещения B;
            scale * cos(angle), 0.f,         -scale * sin(angle), 0.f,  // матрица A имеет вид:      (scale, 0, 0, 0)  - она меняет масштаб по всем                  матрица B имеет вид:
            0.f,                scale * 1.f, 0.f,                 0.f,  //                           (0, scale, 0, 0)    кооррдинатам (x, y и z) сразу                   (R, h)
            scale * sin(angle), 0.f,         scale * cos(angle),  0.f,  //                       A = (0, 0, scale, 0)                                                B = (0, 1), где R - обычная 3 на 3 матрица поворота, h - это 3 на 1 вектор смещения
            0.f,                0.f,         0.f,                 1.f,  //                           (0, 0, 0,     1)
        };
        */

        // === Задание 5 ===
        float angle = time, speed = 1.0;
        if (button_down[SDLK_LEFT])  // в зависимости от нажатых кнопок, меняем сдвиг центра зайца
            bunny_x -= dt * speed;
        if (button_down[SDLK_RIGHT])
            bunny_x += dt * speed;
        if (button_down[SDLK_UP])
            bunny_y += dt * speed;
        if (button_down[SDLK_DOWN])
            bunny_y -= dt * speed;
        float scale = 0.5;
        float model[16] =
        {                                                              
            scale * cos(angle), 0.f,         -scale * sin(angle), bunny_x,  // составляем матрицу, которая теперь помимо поворота как раньше, деает ещё и сдвиг (сначала поворот, затем сдвиг)
            0.f,                scale * 1.f, 0.f,                 bunny_y,  // (всё это относительно исходной системы координат, в которой заяц задан изначально в файле)
            scale * sin(angle), 0.f,         scale * cos(angle),  0.f,
            0.f,                0.f,         0.f,                 1.f,
        };


        // === Задание 7 ===
        float model_xy[16] =
        {                                                              
            scale * cos(angle), -scale * sin(angle), 0.f,         -1.2,
            scale * sin(angle), scale * cos(angle),  0.f,         0.5,
            0.f,                0.f,                 scale * 1.f, 0.f,
            0.f,                0.f,         0.f,                 1.f,
        };  // добавляем матрицу для зайца, который будет вращаться в плоскости XY (от предыдущего, который вращался в плоскости XZ, отличается только матрицей поворота... и еще смещение bunny_x,bunny_y убираем, чтобы двигался только предыдущий заяц от кнопок... но небольшое смещение ставим (на глаз, подбором), чтобы зайцы в разных углах были)
        float model_yz[16] =
        {                                                              
            scale * 1.f, 0.f,                0.f,                 1.2,
            0.f,         scale * cos(angle), -scale * sin(angle), -0.5,
            0.f,         scale * sin(angle), scale * cos(angle),  0.f,
            0.f,                0.f,         0.f,                 1.f,
        };  // для зайца в плоскости YZ


        /*
        float view[16] =
        {
            1.f, 0.f, 0.f, 0.f,
            0.f, 1.f, 0.f, 0.f,
            0.f, 0.f, 1.f, 0.f,
            0.f, 0.f, 0.f, 1.f,
        };
        */

        // === Задание 4 ===
        float view[16] =
        {
            1.f, 0.f, 0.f, 0.f,
            0.f, 1.f, 0.f, 0.f,
            0.f, 0.f, 1.f, -2.f,  // сдвигаем камеру по z на несколько единиц (в остальном - оставляем единичную матрицу)
            0.f, 0.f, 0.f, 1.f,
        };

        /*
        float projection[16] =
        {
            1.f, 0.f, 0.f, 0.f,
            0.f, 1.f, 0.f, 0.f,
            0.f, 0.f, 1.f, 0.f,
            0.f, 0.f, 0.f, 1.f,
        };
        */

        // === Задание 3 ===
        float near = 0.1, far = 100, theta = 90;  // значения near, far, а также угла обзора (в градусах)
        float right = near * tan(theta * M_PIf / 180 / 2);  // near умножаем на тангенс половинного угла обзора (переводим его в радианы сначала)
        float top = right * height / width;  // вычисляем top через aspect ratio экрана
        float projection[16] =
        {
            near/right, 0.f,      0.f,                    0.f,
            0.f,        near/top, 0.f,                    0.f,
            0.f,        0.f,      -(far+near)/(far-near), -2*far*near/(far-near),
            0.f,        0.f,      -1.f,                   0.f,
        };  // матрица из лекции


        glUseProgram(program);
        glUniformMatrix4fv(model_location, 1, GL_TRUE, model);
        glUniformMatrix4fv(view_location, 1, GL_TRUE, view);
        glUniformMatrix4fv(projection_location, 1, GL_TRUE, projection);


        // === Задание 1 ===
        glBindVertexArray(vao);
        glDrawElements(GL_TRIANGLES, bunny.indices.size(), GL_UNSIGNED_INT, (void*)(0));


        // === Задание 7 ===
        glUniformMatrix4fv(model_location, 1, GL_TRUE, model_xy);  // устанавливаем матрицу и отрисовываем двух оставшихся зайцев
        glDrawElements(GL_TRIANGLES, bunny.indices.size(), GL_UNSIGNED_INT, (void*)(0));
        glUniformMatrix4fv(model_location, 1, GL_TRUE, model_yz);
        glDrawElements(GL_TRIANGLES, bunny.indices.size(), GL_UNSIGNED_INT, (void*)(0));


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
