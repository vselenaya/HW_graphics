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
#include <fstream>
#include <chrono>
#include <vector>
#include <random>
#include <map>
#include <cmath>

#include <glm/vec3.hpp>
#include <glm/mat4x4.hpp>
#include <glm/ext/matrix_transform.hpp>
#include <glm/ext/matrix_clip_space.hpp>
#include <glm/ext/scalar_constants.hpp>
#include <glm/gtx/quaternion.hpp>
#include <glm/gtx/string_cast.hpp>

#include "msdf_loader.hpp"
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

/* было изначально
const char msdf_vertex_shader_source[] =
R"(#version 330 core

uniform mat4 transform;

void main()
{
    gl_Position = vec4(0.0, 0.0, 0.0, 1.0);
}
)";

const char msdf_fragment_shader_source[] =
R"(#version 330 core

layout (location = 0) out vec4 out_color;

void main()
{
    out_color = vec4(0.0);
}
)";
*/

/*
// === Задание 1 ===
const char msdf_vertex_shader_source[] =
R"(#version 330 core

layout(location = 0) in vec2 position;  // добавили аттрибуты вершин
layout(location = 1) in vec2 texcoord;

out vec2 frag_texcoord;

uniform mat4 transform;

void main()
{
    gl_Position = vec4(position, 0.0, 1.0);
    frag_texcoord = texcoord;  // прокидываем текстурные во фрагментный шейдер
}
)";
*/


// === Задание 2 ===
const char msdf_vertex_shader_source[] =
R"(#version 330 core

layout(location = 0) in vec2 position;
layout(location = 1) in vec2 texcoord;

out vec2 frag_texcoord;

uniform mat4 transform;

void main()
{
    gl_Position = transform * vec4(position, 0.0, 1.0);  // применяем матрицу
    frag_texcoord = texcoord;
}
)";


/*
// === Задание 1 ===
const char msdf_fragment_shader_source[] =
R"(#version 330 core

in vec2 frag_texcoord;  // принимаем коорднаты из вершнного шейдера
layout (location = 0) out vec4 out_color;

void main()
{
    out_color = vec4(frag_texcoord, 0.0, 1.0);  // цвет такой
}
)";
*/

/*
// === Задание 4 ===
const char msdf_fragment_shader_source[] =
R"(#version 330 core

in vec2 frag_texcoord;
layout (location = 0) out vec4 out_color;

uniform sampler2D sdf_texture;  // Текстура шрифта
uniform float sdf_scale;  // Масштаб SDF

float median(vec3 v) {
    return max(min(v.r, v.g), min(max(v.r, v.g), v.b));
}

void main()
{
    // Получаем значение из текстуры
    float texture_value = median(texture(sdf_texture, frag_texcoord).rgb);  // Вычисляем как в 39 слайде лекции

    // Применяем SDF для вычисления альфа-канала
    float sdf_value = sdf_scale * (texture_value - 0.5);  // сглаживание
    float alpha = smoothstep(-0.5, 0.5, sdf_value);  

    out_color = vec4(0.0, 0.0, 0.0, alpha);  // Чёрный цвет с вычисленной прозрачностью
}
)";
*/

/*
// === Задание 6 ===
const char msdf_fragment_shader_source[] =
R"(#version 330 core

in vec2 frag_texcoord;
layout (location = 0) out vec4 out_color;

uniform sampler2D sdf_texture;  // Текстура шрифта
uniform float sdf_scale;  // Масштаб SDF

float median(vec3 v) {
    return max(min(v.r, v.g), min(max(v.r, v.g), v.b));
}

void main()
{
    float texture_value = median(texture(sdf_texture, frag_texcoord).rgb); 

    float sdf_value = sdf_scale * (texture_value - 0.5);
    float val = length(vec2(dFdx(sdf_value), dFdy(sdf_value))) / sqrt(2.0);
    float alpha = smoothstep(-val, val, sdf_value);  // меняем на другую величину!  

    out_color = vec4(0.0, 0.0, 0.0, alpha); 
}
)";
*/


// === Задание 7 ===
const char msdf_fragment_shader_source[] =
R"(#version 330 core

in vec2 frag_texcoord;
layout (location = 0) out vec4 out_color;

uniform sampler2D sdf_texture;
uniform float sdf_scale;

float median(vec3 v) {
    return max(min(v.r, v.g), min(max(v.r, v.g), v.b));
}

void main()
{
    vec3 text_color = vec3(0.0, 0.0, 0.0);  // цвет текста
    vec3 outline_color = vec3(1.0, 1.0, 1.0);  // цвет обводки
    float eps = 1;  // толщина обводки: при sdf > 0 рисуем букву (пиксели цвета text_color)
                    //                  при sdf \in [-eps, 0] рисуем обводку (пиксели цвета outline_color)
                    //                  при sdf < -eps ничего (прозрачные пиксели)

    float texture_value = median(texture(sdf_texture, frag_texcoord).rgb); 
    float sdf_value = sdf_scale * (texture_value - 0.5);  // считываем значение из тектуры и получаем по нему реально значение sdf

    float val = length(vec2(dFdx(sdf_value), dFdy(sdf_value))) / sqrt(2.0);  // хорошо подобранное значение, чтобы не было алиасинг
    // smoothstep(a, b, f) даёт 0, если f < a; 1, если f > b; и число от 0 до 1 (гадко интерполирует), если f от a до b
    float mixed = smoothstep(-val, val, sdf_value);  // сначала получаем коэффициент в окрестности (толщины +-val) значения sdf_value = 0 (это граница
                                                     // между буквой и обводкой -> с этим коэффициентом будем смешивать цвета, чтобы был гладкий переход от буквы к обводке)
    float alpha = smoothstep(-val, val, sdf_value+eps);  // теперь сдвигаем весь sdf на eps и снова получаем коэффициент для той же окрестности +-val ->
                                                         // -> этот коэффициент уже соответсвует sdf_value = -eps и используется нами гладкой границы обводки (этот коэффициент = переход от обводки к прозрачным пикселям снаружи)
    
    out_color = vec4(mix(outline_color, text_color, mixed), alpha); 
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

template <typename ... Shaders>
GLuint create_program(Shaders ... shaders)
{
    GLuint result = glCreateProgram();
    (glAttachShader(result, shaders), ...);
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



// === Задание 1 ===
struct Vertex {  // структура вершины
    glm::vec2 position;
    glm::vec2 texcoord;
};



int main() try
{
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

    SDL_Window * window = SDL_CreateWindow("Graphics course practice 15",
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

    auto msdf_vertex_shader = create_shader(GL_VERTEX_SHADER, msdf_vertex_shader_source);
    auto msdf_fragment_shader = create_shader(GL_FRAGMENT_SHADER, msdf_fragment_shader_source);
    auto msdf_program = create_program(msdf_vertex_shader, msdf_fragment_shader);

    GLuint transform_location = glGetUniformLocation(msdf_program, "transform");

    const std::string project_root = PROJECT_ROOT;
    const std::string font_path = project_root + "/font/font-msdf.json";

    auto const font = load_msdf_font(font_path);

    GLuint texture;
    int texture_width, texture_height;
    {
        int channels;
        auto data = stbi_load(font.texture_path.c_str(), &texture_width, &texture_height, &channels, 4);
        assert(data);

        glGenTextures(1, &texture);
        glBindTexture(GL_TEXTURE_2D, texture);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_LINEAR);
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, texture_width, texture_height, 0, GL_RGBA, GL_UNSIGNED_BYTE, data);
        glGenerateMipmap(GL_TEXTURE_2D);

        stbi_image_free(data);
    }

    auto last_frame_start = std::chrono::high_resolution_clock::now();

    float time = 0.f;

    SDL_StartTextInput();

    std::map<SDL_Keycode, bool> button_down;

    std::string text = "Hello, world!";
    bool text_changed = true;




    // ====== МОЙ КОД: ======

    // === Задание 1 ===
    GLuint vao, vbo;
    std::vector<Vertex> vertices = {
        {{0.0f, 0.0f}, {0.0f, 0.0f}}, // Вершина 1 (координаты + текстурные)
        {{100.0f, 0.0f}, {1.0f, 0.0f}}, // Вершина 2
        {{0.0f, 100.0f}, {0.0f, 1.0f}}  // Вершина 3
    };

    glGenVertexArrays(1, &vao);
    glBindVertexArray(vao);

    glGenBuffers(1, &vbo);  // создаём и заполняем vbo
    glBindBuffer(GL_ARRAY_BUFFER, vbo);
    glBufferData(GL_ARRAY_BUFFER, vertices.size() * sizeof(Vertex), vertices.data(), GL_STATIC_DRAW);

    glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, sizeof(Vertex), (void*)offsetof(Vertex, position));  // Настройка атрибутов - для position и texcoord
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, sizeof(Vertex), (void*)offsetof(Vertex, texcoord));
    glEnableVertexAttribArray(1);



    // === Задание 3 ===
    size_t vertex_count = 0;  // количество вершин для рисования глифа



    // === Задание 5 ===
    float min_x = std::numeric_limits<float>::max();  // bounding box текста
    float min_y = std::numeric_limits<float>::max();
    float max_x = std::numeric_limits<float>::lowest();
    float max_y = std::numeric_limits<float>::lowest();



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
            if (event.key.keysym.sym == SDLK_BACKSPACE && !text.empty())
            {
                text.pop_back();
                text_changed = true;
            }
            break;
        case SDL_TEXTINPUT:
            text.append(event.text.text);
            text_changed = true;
        case SDL_KEYUP:
            button_down[event.key.keysym.sym] = false;
            break;
        }

        if (!running)
            break;

        auto now = std::chrono::high_resolution_clock::now();
        float dt = std::chrono::duration_cast<std::chrono::duration<float>>(now - last_frame_start).count();
        last_frame_start = now;

        glClearColor(0.8f, 0.8f, 1.f, 0.f);
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

        glEnable(GL_BLEND);
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

        glActiveTexture(GL_TEXTURE0);
        glBindTexture(GL_TEXTURE_2D, texture);



        // === Задание 2 ===
        glm::mat4 transform = glm::mat4(1.0f); // Инициализация единичной матрицы, далее настраиваем её (замеим, что в основнм цикле деалаем, так как width, height могут поменяться при растяжении окна)
        transform = glm::scale(transform, glm::vec3(1.0f, -1.0f, 1.0f)); // Инвертирование оси Y
        transform = glm::scale(transform, glm::vec3(1.0f / (width / 2.0f), 1.0f / (height / 2.0f), 1.0f)); // Масштабирование
        transform = glm::translate(transform, glm::vec3(-width / 2.f, -height / 2.f, 0.0f)); // Перемещение по осям -> теперь координаты будут x \in [-width/2, width/2] и y \in [-height/2, height/2]
        //auto v3 = glm::vec4(width, height, 0, 1);
        //v3 = transform * v3;
        //std::cout << v3.x << " " << v3.y << " " << v3.z << std::endl;  // это должно выдавать: 1 -1 0
        glUseProgram(msdf_program);
        GLuint transform_location = glGetUniformLocation(msdf_program, "transform");  // получаем местоположение и переадём в неё transform
        glUniformMatrix4fv(transform_location, 1, GL_FALSE, reinterpret_cast<float *> (&transform));



        // === Задание 3 ===
        if (text_changed) {
            vertices.clear();  // Очищаем вектор вершин
            auto pen = glm::vec2(0.0f);  // Сбрасываем позицию пера

            for (char c: text) {
                // Получаем глиф для текущего символа
                const auto& glyph = font.glyphs.at(c);  // получаем из map значение по ключу c через .at(), а не [] (так как константное занчение!)

                // Вычисляем координаты вершин
                float x0 = pen.x + glyph.xoffset;
                float y0 = pen.y + glyph.yoffset;
                float x1 = x0 + glyph.width;
                float y1 = y0 + glyph.height;

                // Вычисляем текстурные координаты
                float u0 = glyph.x * 1.f / texture_width;  // не заюываем  * 1.f, чтобы привести значения float (чтобы деление было с плавающей точко, не нацело)
                float v0 = glyph.y * 1.f / texture_height;
                float u1 = (glyph.x + glyph.width) * 1.f / texture_width;
                float v1 = (glyph.y + glyph.height) * 1.f / texture_height;

                // Добавляем 6 вершин для двух треугольников
                vertices.push_back({{x0, y0}, {u0, v0}});
                vertices.push_back({{x1, y0}, {u1, v0}});
                vertices.push_back({{x0, y1}, {u0, v1}});
                vertices.push_back({{x1, y0}, {u1, v0}});
                vertices.push_back({{x1, y1}, {u1, v1}});
                vertices.push_back({{x0, y1}, {u0, v1}});

                // Сдвигаем перо
                pen.x += glyph.advance;


                // === Задание 5 ===
                min_x = std::min(min_x, x0);  // Обновляем bounding box
                min_y = std::min(min_y, y0);
                max_x = std::max(max_x, x1);
                max_y = std::max(max_y, y1);

            }

            // Загружаем новые вершины в VBO
            glBindVertexArray(vao);
            glBindBuffer(GL_ARRAY_BUFFER, vbo);
            glBufferData(GL_ARRAY_BUFFER, vertices.size() * sizeof(Vertex), vertices.data(), GL_STATIC_DRAW);

            // Обновляем количество вершин
            vertex_count = vertices.size();

            // Сбрасываем флаг
            text_changed = false;
        }



        // === Задание 4 ===
        GLuint sdf_scale_location = glGetUniformLocation(msdf_program, "sdf_scale");
        glUniform1f(sdf_scale_location, font.sdf_scale); // Передаем значение sdf_scale в шейдер



        // === Задание 5 ===
        float center_x = (min_x + max_x) / 2.f;  // Вычисляем центр текста (в координатах экрана [0, width] на [0, height])
        float center_y = (min_y + max_y) / 2.f;
        transform = glm::mat4(1.0f);  // Аналогично заданию 2 считаем матрицу transform (а именно домножаем её (инициализированную как единичную) справа на различные матрицы: сначала два раза на scale, затем одн раз на translate - именно в таком порядке, так как мы хотим, чтобы translate шёл самым первым -> он идёт последней строчкой в настройке transform (так как домножения идут справа, но и применение матрицы к вектору так же идёт -> чем на более поздней строке применим преобразование к transform, ем раньше оно применится к вектору))
        transform = glm::scale(transform, glm::vec3(1.0f, -1.0f, 1.0f));
        transform = glm::scale(transform, glm::vec3(1.0f / (width / 2.0f), 1.0f / (height / 2.0f), 1.0f));
        transform = glm::translate(transform, glm::vec3(-center_x, -center_y, 0.0f));  // в самой поздней строке настройки transform (= значит самым ранним преобразованием вектр postion, к которому transform применяется) делаем сдвиг -> после этого диапозон координта двигается: [0, width] × [0, height] ---> [-center_x, width - center_x] × [-center_y, height - center_y]
                                                                                       // (то есть тут в точности, как в задании 2 - только если там делали сдвиг на (-w/2, -h/2), чтобы (0,0) (в координатах экрана [0,w]×[0,h]) перешёл в (-1, 1) (в кооординатах openg [-1,1]×[-1,1]), то тут делаем такой свдиг, чтобы центр текста перешёл в (0,0) координата opengl [-1,1]×[-1,1])
        glUseProgram(msdf_program);
        glUniformMatrix4fv(transform_location, 1, GL_FALSE, reinterpret_cast<float *> (&transform));



        // === Задание 6 ===
        transform = glm::mat4(1.0f);
        transform = glm::scale(transform, glm::vec3(1.0f, -1.0f, 1.0f));
        transform = glm::scale(transform, glm::vec3(1.0f / (width / 2.0f), 1.0f / (height / 2.0f), 1.0f));
        transform = glm::scale(transform, glm::vec3(5.0f, 5.0f, 1.0f));  // Снова пересчитываем transform: теперь (после сдвига! который на следующей строке) ещё увеличиваем масштаб в 5 раз
        transform = glm::translate(transform, glm::vec3(-center_x, -center_y, 0.0f));
        glUseProgram(msdf_program);
        glUniformMatrix4fv(transform_location, 1, GL_FALSE, reinterpret_cast<float *> (&transform));



        /*
        // === Задание 1 ===
        glUseProgram(msdf_program);
        glBindVertexArray(vao);
        glDrawArrays(GL_TRIANGLES, 0, 3);  // рисуем треугольник (три вершины)
        */



        // === Задание 3 ===
        glUseProgram(msdf_program);
        glBindVertexArray(vao);
        glDrawArrays(GL_TRIANGLES, 0, vertex_count);  // аналогично здаанию 1 рисуем, но теперь с нужным количество вершин



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
