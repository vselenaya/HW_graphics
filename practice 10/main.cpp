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

#define GLM_FORCE_SWIZZLE
#define GLM_ENABLE_EXPERIMENTAL
#include <glm/vec3.hpp>
#include <glm/mat4x4.hpp>
#include <glm/ext/matrix_transform.hpp>
#include <glm/ext/matrix_clip_space.hpp>
#include <glm/ext/scalar_constants.hpp>
#include <glm/gtx/string_cast.hpp>

#include "obj_parser.hpp"
#include "stb_image.h"

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
layout (location = 1) in vec3 in_tangent;
layout (location = 2) in vec3 in_normal;
layout (location = 3) in vec2 in_texcoord;

out vec3 position;
out vec3 tangent;
out vec3 normal;
out vec2 texcoord;

void main()
{
    position = (model * vec4(in_position, 1.0)).xyz;
    gl_Position = projection * view * vec4(position, 1.0);
    tangent = mat3(model) * in_tangent;
    normal = mat3(model) * in_normal;
    texcoord = in_texcoord;
}
)";

/*было изначально
const char fragment_shader_source[] =
R"(#version 330 core

uniform vec3 light_direction;
uniform vec3 camera_position;

uniform sampler2D albedo_texture;

in vec3 position;
in vec3 tangent;
in vec3 normal;
in vec2 texcoord;

layout (location = 0) out vec4 out_color;

const float PI = 3.141592653589793;

void main()
{
    float ambient_light = 0.2;

    float lightness = ambient_light + max(0.0, dot(normalize(normal), light_direction));

    vec3 albedo = texture(albedo_texture, texcoord).rgb;

    out_color = vec4(lightness * albedo, 1.0);
}
)";
*/

/*
// === Задание 1 ===
const char fragment_shader_source[] =
R"(#version 330 core

uniform vec3 light_direction;
uniform vec3 camera_position;

uniform sampler2D albedo_texture;

in vec3 position;
in vec3 tangent;
in vec3 normal;
in vec2 texcoord;

layout (location = 0) out vec4 out_color;

const float PI = 3.141592653589793;

void main()
{
    float ambient_light = 0.2;
    float lightness = ambient_light + max(0.0, dot(normalize(normal), light_direction));
    vec3 albedo = normal * 0.5 + vec3(0.5);  // выводим нормаль (её компоненты от -1 до 1 (так как вектор нормированный = длины 1) -> переводим их в [0,1])
    out_color = vec4(lightness * albedo, 1.0);
}
)";
*/

/*
// === Задание 2 ===
const char fragment_shader_source[] =
R"(#version 330 core

uniform vec3 light_direction;
uniform vec3 camera_position;

uniform sampler2D albedo_texture;
uniform sampler2D normal_texture;  // заводим для текстуры

in vec3 position;
in vec3 tangent;
in vec3 normal;
in vec2 texcoord;

layout (location = 0) out vec4 out_color;

const float PI = 3.141592653589793;

void main()
{
    float ambient_light = 0.2;
    float lightness = ambient_light + max(0.0, dot(normalize(normal), light_direction));
    vec3 albedo = texture(normal_texture, texcoord).rgb;  // читаем из текстуры
    out_color = vec4(lightness * albedo, 1.0);
}
)";
*/

/*
// === Задание 3 ===
const char fragment_shader_source[] =
R"(#version 330 core

uniform vec3 light_direction;
uniform vec3 camera_position;

uniform sampler2D albedo_texture;
uniform sampler2D normal_texture;

in vec3 position;
in vec3 tangent;
in vec3 normal;
in vec2 texcoord;

layout (location = 0) out vec4 out_color;

const float PI = 3.141592653589793;

void main()
{
    vec3 bitangent = cross(tangent, normal);  // Во фрагментном шейдере вычисляем bitangent вектор
    mat3 tbn = mat3(tangent, bitangent, normal);  // Вычисляем матрицу преобразования из normal map в мировые координаты
    vec3 real_normal = tbn * (texture(normal_texture, texcoord).xyz * 2.0 - vec3(1.0));  // Прочитанное из normal map значение переводим из [0, 1] в [-1, 1] и применяем матрицу:

    float ambient_light = 0.2;
    float lightness = ambient_light + max(0.0, dot(normalize(normal), light_direction));
    vec3 albedo = real_normal * 0.5 + vec3(0.5);  // получаем цвет (снова переводи от 0 до 1, так как таким цвет должен быть)
    out_color = vec4(lightness * albedo, 1.0);
}
)";
*/

/*
// === Задание 4 ===
const char fragment_shader_source[] =
R"(#version 330 core

uniform vec3 light_direction;
uniform vec3 camera_position;

uniform sampler2D albedo_texture;
uniform sampler2D normal_texture;

in vec3 position;
in vec3 tangent;
in vec3 normal;
in vec2 texcoord;

layout (location = 0) out vec4 out_color;

const float PI = 3.141592653589793;

void main()
{
    vec3 bitangent = cross(tangent, normal);
    mat3 tbn = mat3(tangent, bitangent, normal);
    vec3 real_normal = tbn * (texture(normal_texture, texcoord).xyz * 2.0 - vec3(1.0));

    float ambient_light = 0.2;
    float lightness = ambient_light + max(0.0, dot(normalize(real_normal), light_direction));  // используем real_normal
    vec3 albedo = texture(albedo_texture, texcoord).rgb;
    out_color = vec4(lightness * albedo, 1.0);
}
)";
*/



// === Задание 5 ===
const char fragment_shader_source[] =
R"(#version 330 core

uniform vec3 light_direction;
uniform vec3 camera_position;  // положение камеры

uniform sampler2D albedo_texture;
uniform sampler2D normal_texture;
uniform sampler2D environment_texture;  // текстура для окружения

in vec3 position;  // положение данного (для которого запущен шейдер) пикселя
in vec3 tangent;
in vec3 normal;
in vec2 texcoord;

layout (location = 0) out vec4 out_color;

const float PI = 3.141592653589793;

void main()
{   
    // считаем real_normal как ранее:
    vec3 bitangent = cross(tangent, normal);
    mat3 tbn = mat3(tangent, bitangent, normal);
    vec3 real_normal = tbn * (texture(normal_texture, texcoord).xyz * 2.0 - vec3(1.0));

    // считаем координаты в текстуре для environement:
    vec3 camera_direction = camera_position - position;  // направление в камеру из данного пикселя
    vec3 dir = 2.0 * real_normal * dot(real_normal, camera_direction) - camera_direction;  // отражаем луч (направление на камеру) с помощью нормали (точнее - реальной real_normal) в данном пикселе - то есть dir это отраженное направление на камеру (как если бы данный пиксель с данной нормалью был зеркалом)
    float x = atan(dir.z, dir.x) / PI * 0.5 + 0.5;  // вычисляем координаты (atan возвращает угол от -PI до PI -> переводим это в угол от 0 до 1 - это искомая координата)
    float y = -atan(dir.y, length(dir.xz)) / PI + 0.5;
    
    // считаем цвет пикселя из environment:
    vec3 new_color = texture(environment_texture, vec2(x, y)).rgb;  // читаем по полученным коордианатам

    // считаем цвет пикселя как раньше:
    float ambient_light = 0.2;
    float lightness = ambient_light + max(0.0, dot(normalize(real_normal), light_direction));
    vec3 albedo = texture(albedo_texture, texcoord).rgb;
    vec3 old_color = lightness * albedo;

    // наконец итоговый цвет - среднее между новым и старым:
    out_color = vec4(mix(new_color, old_color, 0.5), 1.0);
}
)";



// === Задание 6 ===
const char env_vertex_source[] = 
R"(#version 330 core

const vec2 VERTICES[6] = vec2[6](  // захардкоженные вершины двух треугольников, образующих прямоугольник на весь экран (координаты +-1 по x и y)
    vec2(-1.0, -1.0),  // три вершины (в порядке против часовой стрелки) первого треугольника (левый верхний)
    vec2(1.0, 1.0),
    vec2(-1.0, 1.0),
    
    vec2(-1.0, -1.0),  // второй треугольник
    vec2(1.0, -1.0),
    vec2(1.0, 1.0)
);

uniform mat4 view;  // матрицы view и projection те же самые, что и для сферы (те же, что изначально в коде были) - так как они просто камеру характеризуют
uniform mat4 projection;

out vec3 position;  // позиция в пространстве точки, изображение которой является вершиной треугольников - это интерполируется и передаётся во фрагментный шейдер далее

void main() {
    vec2 vertex = VERTICES[gl_VertexID];  // просто по индексу берем новую точку из массива
    gl_Position = vec4(vertex, 0.0, 1.0);  // вершина, которую нужно нарисовать
    vec4 ndc = vec4(vertex, 0.0, 1.0);
    vec4 clip_space = inverse(projection * view) * ndc;  // прообраз вершины в пространстве
    position = clip_space.xyz / clip_space.w;
}
)";

const char env_fragment_source[] =
R"(#version 330 core

uniform sampler2D environment_texture;  // текстура, которую рисуем на фоне
uniform vec3 camera_position;  // положение камеры

in vec3 position;  // координаты точки в пространстве из шейдера ранее

layout (location = 0) out vec4 out_color;

const float PI = 3.141592653589793;

void main() {
    vec3 dir = position - camera_position;  // !!! важно - нас тут интересует направдение из камеры в точку (а не как ранее наоборот)
    float theta = atan(dir.z, dir.x) / PI * 0.5 + 0.5;  // аналогично предыдущему выисляем координаты (а именно - широту и долготу точки position относительно камеры)
    float phi = -atan(dir.y, length(dir.xz)) / PI + 0.5;
    out_color = vec4(texture(environment_texture, vec2(theta, phi)).rgb, 1.0);  // используем для получения значения из текстуры
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

struct vertex
{
    glm::vec3 position;
    glm::vec3 tangent;
    glm::vec3 normal;
    glm::vec2 texcoords;
};

std::pair<std::vector<vertex>, std::vector<std::uint32_t>> generate_sphere(float radius, int quality)
{
    std::vector<vertex> vertices;

    for (int latitude = -quality; latitude <= quality; ++latitude)
    {
        for (int longitude = 0; longitude <= 4 * quality; ++longitude)
        {
            float lat = (latitude * glm::pi<float>()) / (2.f * quality);
            float lon = (longitude * glm::pi<float>()) / (2.f * quality);

            auto & vertex = vertices.emplace_back();
            vertex.normal = {std::cos(lat) * std::cos(lon), std::sin(lat), std::cos(lat) * std::sin(lon)};
            vertex.position = vertex.normal * radius;
            vertex.tangent = {-std::cos(lat) * std::sin(lon), 0.f, std::cos(lat) * std::cos(lon)};
            vertex.texcoords.x = (longitude * 1.f) / (4.f * quality);
            vertex.texcoords.y = (latitude * 1.f) / (2.f * quality) + 0.5f;
        }
    }

    std::vector<std::uint32_t> indices;

    for (int latitude = 0; latitude < 2 * quality; ++latitude)
    {
        for (int longitude = 0; longitude < 4 * quality; ++longitude)
        {
            std::uint32_t i0 = (latitude + 0) * (4 * quality + 1) + (longitude + 0);
            std::uint32_t i1 = (latitude + 1) * (4 * quality + 1) + (longitude + 0);
            std::uint32_t i2 = (latitude + 0) * (4 * quality + 1) + (longitude + 1);
            std::uint32_t i3 = (latitude + 1) * (4 * quality + 1) + (longitude + 1);

            indices.insert(indices.end(), {i0, i1, i2, i2, i1, i3});
        }
    }

    return {std::move(vertices), std::move(indices)};
}

GLuint load_texture(std::string const & path)
{
    int width, height, channels;
    auto pixels = stbi_load(path.data(), &width, &height, &channels, 4);

    GLuint result;
    glGenTextures(1, &result);
    glBindTexture(GL_TEXTURE_2D, result);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, width, height, 0, GL_RGBA, GL_UNSIGNED_BYTE, pixels);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glGenerateMipmap(GL_TEXTURE_2D);

    stbi_image_free(pixels);

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

    SDL_Window * window = SDL_CreateWindow("Graphics course practice 5",
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
    GLuint light_direction_location = glGetUniformLocation(program, "light_direction");
    GLuint camera_position_location = glGetUniformLocation(program, "camera_position");
    GLuint albedo_texture_location = glGetUniformLocation(program, "albedo_texture");

    GLuint sphere_vao, sphere_vbo, sphere_ebo;
    glGenVertexArrays(1, &sphere_vao);
    glBindVertexArray(sphere_vao);
    glGenBuffers(1, &sphere_vbo);
    glGenBuffers(1, &sphere_ebo);
    GLuint sphere_index_count;
    {
        auto [vertices, indices] = generate_sphere(1.f, 16);

        glBindBuffer(GL_ARRAY_BUFFER, sphere_vbo);
        glBufferData(GL_ARRAY_BUFFER, vertices.size() * sizeof(vertices[0]), vertices.data(), GL_STATIC_DRAW);

        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, sphere_ebo);
        glBufferData(GL_ELEMENT_ARRAY_BUFFER, indices.size() * sizeof(indices[0]), indices.data(), GL_STATIC_DRAW);

        sphere_index_count = indices.size();
    }
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, sizeof(vertex), (void *)offsetof(vertex, position));
    glEnableVertexAttribArray(1);
    glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, sizeof(vertex), (void *)offsetof(vertex, tangent));
    glEnableVertexAttribArray(2);
    glVertexAttribPointer(2, 3, GL_FLOAT, GL_FALSE, sizeof(vertex), (void *)offsetof(vertex, normal));
    glEnableVertexAttribArray(3);
    glVertexAttribPointer(3, 3, GL_FLOAT, GL_FALSE, sizeof(vertex), (void *)offsetof(vertex, texcoords));

    std::string project_root = PROJECT_ROOT;
    GLuint albedo_texture = load_texture(project_root + "/textures/brick_albedo.jpg");

    auto last_frame_start = std::chrono::high_resolution_clock::now();

    float time = 0.f;

    std::map<SDL_Keycode, bool> button_down;

    float view_elevation = glm::radians(30.f);
    float view_azimuth = 0.f;
    float camera_distance = 2.f;



    // ======= МОЙ КОД ========
    
    // === Задание 2 ===
    GLuint normal_texture = load_texture(project_root + "/textures/brick_normal.jpg");
    GLuint normal_texture_location = glGetUniformLocation(program, "normal_texture");


    // === Задание 5 ===
    GLuint environment_texture = load_texture(project_root + "/textures/environment_map.jpg");
    GLuint environment_texture_location = glGetUniformLocation(program, "environment_texture");

    
    // === Задание 6 ===
    auto env_vertex_shader = create_shader(GL_VERTEX_SHADER, env_vertex_source);
    auto env_fragment_shader = create_shader(GL_FRAGMENT_SHADER, env_fragment_source);
    auto env_program = create_program(env_vertex_shader, env_fragment_shader);

    GLuint env_texture_location = glGetUniformLocation(env_program, "environment_texture");
    GLuint env_view_location = glGetUniformLocation(env_program, "view");
    GLuint env_projection_location = glGetUniformLocation(env_program, "projection");
    GLuint env_camera_location = glGetUniformLocation(env_program, "camera_position");

    GLuint empty_vao;
    glGenVertexArrays(1, &empty_vao);  // фиктивный vao

    

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

        if (button_down[SDLK_UP])
            camera_distance -= 4.f * dt;
        if (button_down[SDLK_DOWN])
            camera_distance += 4.f * dt;

        if (button_down[SDLK_LEFT])
            view_azimuth -= 2.f * dt;
        if (button_down[SDLK_RIGHT])
            view_azimuth += 2.f * dt;

        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

        glEnable(GL_DEPTH_TEST);
        glEnable(GL_CULL_FACE);

        float near = 0.1f;
        float far = 100.f;
        float top = near;
        float right = (top * width) / height;

        glm::mat4 model = glm::rotate(glm::mat4(1.f), time * 0.1f, {0.f, 1.f, 0.f});

        glm::mat4 view(1.f);
        view = glm::translate(view, {0.f, 0.f, -camera_distance});
        view = glm::rotate(view, view_elevation, {1.f, 0.f, 0.f});
        view = glm::rotate(view, view_azimuth, {0.f, 1.f, 0.f});

        glm::mat4 projection = glm::mat4(1.f);
        projection = glm::perspective(glm::pi<float>() / 2.f, (1.f * width) / height, near, far);

        glm::vec3 light_direction = glm::normalize(glm::vec3(1.f, 2.f, 3.f));

        glm::vec3 camera_position = (glm::inverse(view) * glm::vec4(0.f, 0.f, 0.f, 1.f)).xyz();



        // === Задание 6 ===
        glUseProgram(env_program);  // не забываем обязательно включить нашу с шейдерами программу
        glUniformMatrix4fv(env_view_location, 1, GL_FALSE, reinterpret_cast<float *>(&view));
        glUniformMatrix4fv(env_projection_location, 1, GL_FALSE, reinterpret_cast<float *>(&projection));
        glUniform3fv(env_camera_location, 1, reinterpret_cast<float *>(&camera_position));  // передали значения uniform
        glActiveTexture(GL_TEXTURE2); 
        glBindTexture(GL_TEXTURE_2D, environment_texture);
        glUniform1i(env_texture_location, 2);  // аналогично предыдущим заданиям используем текстуру в шейдере
        glDisable(GL_DEPTH_TEST);  // рисуем фон БЕЗ теста глубины (как в лекциях написано)
        glBindVertexArray(empty_vao);
        glDrawArrays(GL_TRIANGLES, 0, 6);  // рисуем треугольники их 6 вершин, начиная с 0-ой (то есть индекс gl_VertexID, который передастся в вершинный шейдер будет от 0 до 5)
        glEnable(GL_DEPTH_TEST);  // обратно включаем



        glUseProgram(program);
        glUniformMatrix4fv(model_location, 1, GL_FALSE, reinterpret_cast<float *>(&model));
        glUniformMatrix4fv(view_location, 1, GL_FALSE, reinterpret_cast<float *>(&view));
        glUniformMatrix4fv(projection_location, 1, GL_FALSE, reinterpret_cast<float *>(&projection));
        glUniform3fv(light_direction_location, 1, reinterpret_cast<float *>(&light_direction));
        glUniform3fv(camera_position_location, 1, reinterpret_cast<float *>(&camera_position));
        glUniform1i(albedo_texture_location, 0);
    
        glActiveTexture(GL_TEXTURE0);
        glBindTexture(GL_TEXTURE_2D, albedo_texture);


        // === Задание 2 ===
        glActiveTexture(GL_TEXTURE1);  // переключаемся на 1-ый texture unit 
        glBindTexture(GL_TEXTURE_2D, normal_texture);  // в нем делаем текущей нашу текстуру
        glUseProgram(program);
        glUniform1i(normal_texture_location, 1);  // передаем в шейдер номер texture unit (в данном случае = 1)-> шейдер будет брать значения из
                                                  // текстуры, текущей в данном texture unit - то есть из нашей normal_texture

        // === Задание 5 ===
        glActiveTexture(GL_TEXTURE2);  // анлогично для environamnet делаем со 2-ым texture unit
        glBindTexture(GL_TEXTURE_2D, environment_texture); 
        glUseProgram(program);
        glUniform1i(environment_texture_location, 2);


        glBindVertexArray(sphere_vao);
        glDrawElements(GL_TRIANGLES, sphere_index_count, GL_UNSIGNED_INT, nullptr);

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
