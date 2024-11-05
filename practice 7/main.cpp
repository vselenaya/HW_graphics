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
#include <cmath>
#include <fstream>
#include <sstream>

#define GLM_FORCE_SWIZZLE
#define GLM_ENABLE_EXPERIMENTAL

#include <glm/vec3.hpp>
#include <glm/mat4x4.hpp>
#include <glm/ext/matrix_transform.hpp>
#include <glm/ext/matrix_clip_space.hpp>
#include <glm/ext/scalar_constants.hpp>
#include <glm/gtx/string_cast.hpp>

#include "obj_parser.hpp"

std::string to_string(std::string_view str) {
    return std::string(str.begin(), str.end());
}

void sdl2_fail(std::string_view message) {
    throw std::runtime_error(to_string(message) + SDL_GetError());
}

void glew_fail(std::string_view message, GLenum error) {
    throw std::runtime_error(to_string(message) + reinterpret_cast<const char *>(glewGetErrorString(error)));
}

const char vertex_shader_source[] =
        R"(#version 330 core

uniform mat4 model;
uniform mat4 view;
uniform mat4 projection;

layout (location = 0) in vec3 in_position;
layout (location = 1) in vec3 in_normal;

out vec3 position;
out vec3 normal;

void main()
{
    position = (model * vec4(in_position, 1.0)).xyz;
    gl_Position = projection * view * vec4(position, 1.0);
    normal = normalize(mat3(model) * in_normal);
}
)";

/*было изначально
const char fragment_shader_source[] =
        R"(#version 330 core

uniform vec3 camera_position;

uniform vec3 albedo;

uniform vec3 ambient_light;

in vec3 position;
in vec3 normal;

layout (location = 0) out vec4 out_color;

void main()
{
    vec3 ambient = albedo * ambient_light;
    vec3 color = ambient;
    out_color = vec4(color, 1.0);
}
)";
*/

/*
// === Задание 2 ===
const char fragment_shader_source[] =
        R"(#version 330 core

uniform vec3 camera_position;
uniform vec3 albedo;
uniform vec3 ambient_light;  // освещение, которое есть везде (типо от неба)
uniform vec3 sun_direction;  // добавилось направление на солнце (из любой точки объекта оно одинаково, так как Солнце типо бесконечно далеко)
uniform vec3 sun_color;  // и его цвет

in vec3 position;
in vec3 normal;

layout (location = 0) out vec4 out_color;

vec3 diffuse(vec3 direction) {
    return albedo * max(0.0, dot(normal, direction));  // функция для вычисления силы освещения (по факту цвета, который получается из-за освещения) в данном пикселе (от которого вызывается шейдер),
                                                       // при условии, что норамаль к поверхности в данном пикселе (интерполированная между вершин треугольника) равна normal, а направление на источник direction
                                                       // (направление на камеру не важно, так как это диффузионное освещение - одинаковое во все стороны); ещё есть albedo, характеризующее цвет объекта (точнее - как он реагирует на свет: в каком цвете больше сила (значение данной функции) -> такого цвета и будет)
}

void main()
{
    vec3 ambient = albedo * ambient_light;  
    vec3 color = ambient + diffuse(sun_direction) * sun_color;  // цвет, который мы видим, получается из вклада ambient и освещения от солнца (пока и то, и то не зависит от направления на камеру)
    out_color = vec4(color, 1.0);
}
)";
*/


/*
// === Задание 3 ===
const char fragment_shader_source[] =
        R"(#version 330 core

uniform vec3 camera_position;
uniform vec3 albedo;
uniform vec3 ambient_light;
uniform vec3 sun_direction;
uniform vec3 sun_color;
uniform vec3 point_light_position;  // позиция точечного источника света (меняется со временем) - записана кажется в мировых координатах (мы вообще только в них и работаем, а они уже как-то в opengl [-1, 1] строятся) - как и camera
uniform vec3 point_light_color;  // цвет источника
uniform vec3 point_light_attenuation;  // коэффициенты затухания C0, C1, C2 (в лекции было: считаем, что свет затухает по формуле 1/(CO + C1 r + C2 r^2) r - расстояние)

in vec3 position;  // судя по коду вершинного шейдера, это позиция точки (пикселя - интерполируем по вершинам треугольника) в мировой системе координат
in vec3 normal;

layout (location = 0) out vec4 out_color;

vec3 diffuse(vec3 direction) {
    return albedo * max(0.0, dot(normal, direction));  // то есть эта функция получает, как данная точка реагирует на свет из направления direction (получает сразу вектор из 3 значений - коэффициент "восприятия" каждого цвета r, g, b -> чем больше каждый
                                                       // из этих коэффициентов, тем сильнее данный цвет данная точка отражает (точнее, распространяет во все стороны - это же диффузионное освещение) -> тем выраженнее данный цветовой канал виден наблюдателю среди цвета всего объекта) -> это зависит от альбедо (как сам материал реагирует на цветовые каналы) на (косинус) угол налона площадки (это просто коэффициент меняющий яркость)
}

void main()
{
    vec3 point_light_direction = normalize(point_light_position - position);  // получаем направление на точечный источник из данной точки position объекта (пиксель которой рисуем), не забываем нормализовать встроенной функцией
    float point_light_dist = length(point_light_position - position);  // также расстояние получем (= длина вектора (point_light_position - position) из position в точечный источник)
    float coef = 1 / (point_light_attenuation.x + point_light_attenuation.y * point_light_dist + point_light_attenuation.z * point_light_dist * point_light_dist);  // коэффициент затухания света от источника (у Солнца его не было, тк бескончено удаленный источник - у него свет не меняется от микроскопических (по масштаба расстояния до Солнца) изменений у нас в сцене)

    vec3 ambient = albedo * ambient_light;  
    vec3 diff_sun = diffuse(sun_direction) * sun_color;  // часть освещения от солнца (просто умножаем коэффициент на сам свет) - уже было
    vec3 diff_point = diffuse(point_light_direction) * coef * point_light_color;  // аналогично получаем освещенность (цвет) в данном пикселе, благодаря точечному источнику + еще умножаем на коэффициент затухания
    vec3 color = ambient + diff_sun + diff_point;  // итоговый цвет точки - все еще не зависит от направления на камеру... 
    out_color = vec4(color, 1.0);
}
)";
*/

/*
// === Задание 4 ===
const char fragment_shader_source[] =
        R"(#version 330 core

uniform vec3 camera_position;
uniform vec3 albedo;
uniform vec3 ambient_light;
uniform vec3 sun_direction;
uniform vec3 sun_color;
uniform vec3 point_light_position;  
uniform vec3 point_light_color;
uniform vec3 point_light_attenuation; 
uniform float glossiness;  // сила отражения
uniform float roughness;  // 'шершавость' поверхности

in vec3 position;
in vec3 normal;

layout (location = 0) out vec4 out_color;

vec3 diffuse(vec3 direction) {
    return albedo * max(0.0, dot(normal, direction));
}

vec3 specular(vec3 direction) {
    float cosine = dot(normal, direction);  // косинус падения света от источника в направлении direction
    vec3 reflected = 2.0 * normal * cosine - direction;  // направление, куда отразится луч, пришедший из направления direction
    vec3 view_direction = normalize(camera_position - position);  // направлени на камеру (наблюдателя) - вот тут оно наконец важно! так как отражение в разны стороны с разной интенсивностью будет
    float power = 1 / (roughness * roughness) - 1;  // получаем степень по формуле из лекци (слайд 67)
    return glossiness * albedo * pow(max(0.0, dot(reflected, view_direction)), power);  // получаем силу (в виде коэффициента, опть же для каждого из цветов r, g, b), с которой происходит отражение света, пришедшего из направления direction; 
                                                                                        // вот тут уже направление на камеру важно (и альбедо важно - в данном случае то же самое, что и раньше, но могло бы быть и другое - тут оно уже просто коэффициент
                                                                                        // для отражения цвета из данного направления - почти как и раньше, только тогда было распространение во все стороны, а не отражение)
}

void main()
{
    vec3 point_light_direction = normalize(point_light_position - position); 
    float point_light_dist = length(point_light_position - position);
    float coef = 1 / (point_light_attenuation.x + point_light_attenuation.y * point_light_dist + point_light_attenuation.z * point_light_dist * point_light_dist);

    vec3 ambient = albedo * ambient_light;  
    vec3 diff_sun = (diffuse(sun_direction) + specular(sun_direction)) * sun_color;  // учитываем теперь еще и коэффициент из-за отражения (он меняется из-за направления на камеру, в отличие от diffusion)
    vec3 diff_point = (diffuse(point_light_direction) + specular(point_light_direction)) * coef * point_light_color;
    vec3 color = ambient + diff_sun + diff_point;
    out_color = vec4(color, 1.0);
}
)";
*/


// === Задание 5 ===
const char fragment_shader_source[] =
        R"(#version 330 core

uniform vec3 camera_position;
uniform vec3 albedo;
uniform vec3 ambient_light;
uniform vec3 sun_direction;
uniform vec3 sun_color;
uniform vec3 point_light_position;  
uniform vec3 point_light_color;
uniform vec3 point_light_attenuation; 
uniform float glossiness;
uniform float roughness;

in vec3 position;
in vec3 normal;

layout (location = 0) out vec4 out_color;

vec3 diffuse(vec3 direction) {
    return albedo * max(0.0, dot(normal, direction));
}

vec3 specular(vec3 direction) {
    float cosine = dot(normal, direction);
    vec3 reflected = 2.0 * normal * cosine - direction;
    vec3 view_direction = normalize(camera_position - position); 
    float power = 1 / (roughness * roughness) - 1;
    return glossiness * albedo * pow(max(0.0, dot(reflected, view_direction)), power);
}

void main()
{
    vec3 point_light_direction = normalize(point_light_position - position); 
    float point_light_dist = length(point_light_position - position);
    float coef = 1 / (point_light_attenuation.x + point_light_attenuation.y * point_light_dist + point_light_attenuation.z * point_light_dist * point_light_dist);

    vec3 ambient = albedo * ambient_light;  
    vec3 diff_sun = (diffuse(sun_direction) + specular(sun_direction)) * sun_color;
    vec3 diff_point = (diffuse(point_light_direction) + specular(point_light_direction)) * coef * point_light_color;
    vec3 color = ambient + diff_sun + diff_point;
    out_color = vec4(color, 0.5);  // устанавливаем коэффициент прозрачности!
}
)";




GLuint create_shader(GLenum type, const char *source) {
    GLuint result = glCreateShader(type);
    glShaderSource(result, 1, &source, nullptr);
    glCompileShader(result);
    GLint status;
    glGetShaderiv(result, GL_COMPILE_STATUS, &status);
    if (status != GL_TRUE) {
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
    if (status != GL_TRUE) {
        GLint info_log_length;
        glGetProgramiv(result, GL_INFO_LOG_LENGTH, &info_log_length);
        std::string info_log(info_log_length, '\0');
        glGetProgramInfoLog(result, info_log.size(), nullptr, info_log.data());
        throw std::runtime_error("Program linkage failed: " + info_log);
    }

    return result;
}

int main() try {
    if (SDL_Init(SDL_INIT_VIDEO) != 0)
        sdl2_fail("SDL_Init: ");

    SDL_GL_SetAttribute(SDL_GL_CONTEXT_MAJOR_VERSION, 3);
    SDL_GL_SetAttribute(SDL_GL_CONTEXT_MINOR_VERSION, 3);
    SDL_GL_SetAttribute(SDL_GL_CONTEXT_PROFILE_MASK, SDL_GL_CONTEXT_PROFILE_CORE);
    SDL_GL_SetAttribute(SDL_GL_DOUBLEBUFFER, 1);
    SDL_GL_SetAttribute(SDL_GL_RED_SIZE, 8);
    SDL_GL_SetAttribute(SDL_GL_GREEN_SIZE, 8);
    SDL_GL_SetAttribute(SDL_GL_BLUE_SIZE, 8);
    SDL_GL_SetAttribute(SDL_GL_DEPTH_SIZE, 24);

    SDL_Window *window = SDL_CreateWindow("Graphics course practice 7",
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

    glClearColor(0.8f, 0.8f, 1.f, 0.f);

    auto vertex_shader = create_shader(GL_VERTEX_SHADER, vertex_shader_source);
    auto fragment_shader = create_shader(GL_FRAGMENT_SHADER, fragment_shader_source);
    auto program = create_program(vertex_shader, fragment_shader);

    GLuint model_location = glGetUniformLocation(program, "model");
    GLuint view_location = glGetUniformLocation(program, "view");
    GLuint projection_location = glGetUniformLocation(program, "projection");
    GLuint camera_position_location = glGetUniformLocation(program, "camera_position");
    GLuint albedo_location = glGetUniformLocation(program, "albedo");
    GLuint ambient_light_location = glGetUniformLocation(program, "ambient_light");

    std::string project_root = PROJECT_ROOT;
    std::string suzanne_model_path = project_root + "/suzanne.obj";
    obj_data suzanne = parse_obj(suzanne_model_path);

    GLuint suzanne_vao, suzanne_vbo, suzanne_ebo;
    glGenVertexArrays(1, &suzanne_vao);
    glBindVertexArray(suzanne_vao);

    glGenBuffers(1, &suzanne_vbo);
    glBindBuffer(GL_ARRAY_BUFFER, suzanne_vbo);
    glBufferData(GL_ARRAY_BUFFER, suzanne.vertices.size() * sizeof(suzanne.vertices[0]), suzanne.vertices.data(),
                 GL_STATIC_DRAW);

    glGenBuffers(1, &suzanne_ebo);
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, suzanne_ebo);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, suzanne.indices.size() * sizeof(suzanne.indices[0]), suzanne.indices.data(),
                 GL_STATIC_DRAW);

    glEnableVertexAttribArray(0);
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, sizeof(obj_data::vertex), (void *) (0));
    glEnableVertexAttribArray(1);
    glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, sizeof(obj_data::vertex), (void *) (12));

    auto last_frame_start = std::chrono::high_resolution_clock::now();

    float time = 0.f;

    std::map<SDL_Keycode, bool> button_down;

    bool transparent = false;

    float camera_distance = 3.f;
    float camera_x = 0.f;
    float camera_angle = 0.f;
    



    // =========== МОЙ КОД: ===========

    // === Задание 2 ===
    GLuint sun_direction_location = glGetUniformLocation(program, "sun_direction");
    GLuint sun_color_location = glGetUniformLocation(program, "sun_color");
    glUseProgram(program);  // ОБЯЗАТЕЛЬНО делаем так, иначе переменные не установятся
    glUniform3f(sun_direction_location, 0.f, 0.8f, 0.6f);  // вверх (0.8 по y) и вперед (0.6 по z); норма = 0^2 + 0.8^2 + 0.6^2 = 0.64 + 0.36 = 1 - единичный вектор
    glUniform3f(sun_color_location, 1.0, 0.9, 0.8);  // цвет Солнца (кстати, логично: само Солнце белое (r = g = b = 1.0), но атмосфера поглощает часть спектра - особенно синий (потому и небо синее))

    // === Задание 3 ===
    GLuint point_light_position_location = glGetUniformLocation(program, "point_light_position");
    GLuint point_light_color_location = glGetUniformLocation(program, "point_light_color");
    GLuint point_light_attenuation_location = glGetUniformLocation(program, "point_light_attenuation");
    glUniform3f(point_light_color_location, 0.f, 0.8f, 0.3f);  // зелено-синий источник
    glUniform3f(point_light_attenuation_location, 1.0, 0.0, 0.01);

    // === Задание 4 ===
    GLuint glossiness_location = glGetUniformLocation(program, "glossiness");
    GLuint roughness_location = glGetUniformLocation(program, "roughness");
    glUniform1f(glossiness_location, 5.f);
    glUniform1f(roughness_location, 0.1f);  // шершавость мала -> степень большая (из лекции, где мутное зеркало через степень делали) -> поверхность близка к зеркалу

    /*
    bool running = true;
    while (running) {
        for (SDL_Event event; SDL_PollEvent(&event);)
            switch (event.type) {
                case SDL_QUIT:
                    running = false;
                    break;
                case SDL_WINDOWEVENT:
                    switch (event.window.event) {
                        case SDL_WINDOWEVENT_RESIZED:
                            width = event.window.data1;
                            height = event.window.data2;
                            break;
                    }
                    break;
                case SDL_KEYDOWN:
                    button_down[event.key.keysym.sym] = true;
                    if (event.key.keysym.sym == SDLK_SPACE)
                        transparent = !transparent;
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

        if (button_down[SDLK_UP])
            camera_distance -= 4.f * dt;
        if (button_down[SDLK_DOWN])
            camera_distance += 4.f * dt;

        if (button_down[SDLK_LEFT])
            camera_angle += 2.f * dt;
        if (button_down[SDLK_RIGHT])
            camera_angle -= 2.f * dt;

        if (button_down[SDLK_KP_4])
            camera_x -= 4.f * dt;
        if (button_down[SDLK_KP_6])
            camera_x += 4.f * dt;

        glViewport(0, 0, width, height);
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
        glClearColor(0.8f, 0.8f, 1.f, 0.f);

        glEnable(GL_DEPTH_TEST);
        glEnable(GL_CULL_FACE);

        float near = 0.1f;
        float far = 100.f;

        glm::mat4 model(1.f);

        glm::mat4 view(1.f);
        view = glm::translate(view, {0.f, 0.f, -camera_distance});
        view = glm::rotate(view, camera_angle, {0.f, 1.f, 0.f});
        view = glm::translate(view, {-camera_x, 0.f, 0.f});

        float aspect = (float)height / (float)width;
        glm::mat4 projection = glm::perspective(glm::pi<float>() / 3.f, (width * 1.f) / height, near, far);

        glm::vec3 camera_position = (glm::inverse(view) * glm::vec4(0.f, 0.f, 0.f, 1.f)).xyz();

        glUseProgram(program);
        glUniformMatrix4fv(model_location, 1, GL_FALSE, reinterpret_cast<float *>(&model));
        glUniformMatrix4fv(view_location, 1, GL_FALSE, reinterpret_cast<float *>(&view));
        glUniformMatrix4fv(projection_location, 1, GL_FALSE, reinterpret_cast<float *>(&projection));
        glUniform3fv(camera_position_location, 1, (float *) (&camera_position));
        glUniform3f(albedo_location, 0.7f, 0.4f, 0.2f);
        glUniform3f(ambient_light_location, 0.2f, 0.2f, 0.2f);  // Задание 1 видимо уже за нас сделано


        // === Задание 3 ===
        glUniform3f(point_light_position_location, 10 * sin(2 * time), 0.f, 10 * cos(2 * time));  // будем считать, что точечный источник вращается вокруг нуля (мировой системы координат) в плоскости xz (горизонатльная) по радиусу = 10  ("2 * " пишем, чтобы выше скорость)

        // === Задание 5 ===
        if (transparent) {
            glEnable(GL_BLEND);  // включаем смешивание (см лекцию 6)
            glBlendEquation(GL_FUNC_ADD);  // смешивание происходит через сумму 
            glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);  // ставим стандартные коэффициенты как в лекции
        } else  
            glDisable(GL_BLEND);

        glBindVertexArray(suzanne_vao);
        glDrawElements(GL_TRIANGLES, suzanne.indices.size(), GL_UNSIGNED_INT, nullptr);

        SDL_GL_SwapWindow(window);
    }
    */


    // === Задание 6 ===
    bool running = true;
    while (running) {
        for (SDL_Event event; SDL_PollEvent(&event);)
            switch (event.type) {
                case SDL_QUIT:
                    running = false;
                    break;
                case SDL_WINDOWEVENT:
                    switch (event.window.event) {
                        case SDL_WINDOWEVENT_RESIZED:
                            width = event.window.data1;
                            height = event.window.data2;
                            break;
                    }
                    break;
                case SDL_KEYDOWN:
                    button_down[event.key.keysym.sym] = true;
                    if (event.key.keysym.sym == SDLK_SPACE)
                        transparent = !transparent;
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

        if (button_down[SDLK_UP])
            camera_distance -= 4.f * dt;
        if (button_down[SDLK_DOWN])
            camera_distance += 4.f * dt;

        if (button_down[SDLK_LEFT])
            camera_angle += 2.f * dt;
        if (button_down[SDLK_RIGHT])
            camera_angle -= 2.f * dt;

        if (button_down[SDLK_KP_4])
            camera_x -= 4.f * dt;
        if (button_down[SDLK_KP_6])
            camera_x += 4.f * dt;

        glViewport(0, 0, width, height);
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
        glClearColor(0.8f, 0.8f, 1.f, 0.f);

        glEnable(GL_DEPTH_TEST);
        glEnable(GL_CULL_FACE);

        float near = 0.1f;
        float far = 100.f;

        glm::mat4 view(1.f);
        view = glm::translate(view, {0.f, 0.f, -camera_distance});
        view = glm::rotate(view, camera_angle, {0.f, 1.f, 0.f});
        view = glm::translate(view, {-camera_x, 0.f, 0.f});

        float aspect = (float)height / (float)width;
        glm::mat4 projection = glm::perspective(glm::pi<float>() / 3.f, (width * 1.f) / height, near, far);
        glm::vec3 camera_position = (glm::inverse(view) * glm::vec4(0.f, 0.f, 0.f, 1.f)).xyz();

        glUseProgram(program);
        glUniformMatrix4fv(view_location, 1, GL_FALSE, reinterpret_cast<float *>(&view));
        glUniformMatrix4fv(projection_location, 1, GL_FALSE, reinterpret_cast<float *>(&projection));
        glUniform3fv(camera_position_location, 1, (float *) (&camera_position));
        glUniform3f(albedo_location, 0.7f, 0.4f, 0.2f);
        glUniform3f(ambient_light_location, 0.2f, 0.2f, 0.2f);

        glUniform3f(point_light_position_location, 10 * sin(2 * time), 0.f, 10 * cos(2 * time));
        if (transparent) {
            glEnable(GL_BLEND);
            glBlendEquation(GL_FUNC_ADD);
            glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA); 
        } else  
            glDisable(GL_BLEND);
        glBindVertexArray(suzanne_vao);


        // === Задание 6 ===
        for (int i = 0; i < 3; i ++) {  // перебираем строки и столбцы сетки 3 на 3, где рисуем обезьянок
            for (int j = 0; j < 3; j ++) {
                glm::mat4 model(0.3f);  // матрица model отвечает за то, как переводить координаты, в которой задана модель, в мировые координаты сцены -> заводим её пока из 0.3 на диагонали, чтобы уменьшить масштаб (то емть пока мировые коррдинаты отличаются более мелким масштабом от координат объекта, а центры совпадают)
                model = glm::translate(model, {3.f * (j-1), -3.f * (i-1), 0.f});  // делаем model так, чтобы она делала сдвиг (параллел перенос): (i=0,j=0) оказался сдвинут на (x=-3, y=3) - левее и вверх (сдивг ещё по коордиантам помедли кажется... короче подбираем на глаз)

                glUniform1f(glossiness_location, 1 + 3 * i);  // от номера строки меняем от 1 до 7
                glUniform1f(roughness_location, 0.1 + j * 0.2);  // в зависимости от номера столбца roughness от 0.1 до 0.5
                glUniformMatrix4fv(model_location, 1, GL_FALSE, reinterpret_cast<float *>(&model));
                
                glDrawElements(GL_TRIANGLES, suzanne.indices.size(), GL_UNSIGNED_INT, nullptr);
            }
        }
        
        SDL_GL_SwapWindow(window);
    }

    SDL_GL_DeleteContext(gl_context);
    SDL_DestroyWindow(window);
}
catch (std::exception const &e) {
    std::cerr << e.what() << std::endl;
    return EXIT_FAILURE;
}
