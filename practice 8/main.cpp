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

out vec3 position;
out vec3 normal;

void main()
{
    position = (model * vec4(in_position, 1.0)).xyz;
    gl_Position = projection * view * vec4(position, 1.0);
    normal = normalize(mat3(model) * in_normal);
}
)";


/* было изначально
const char fragment_shader_source[] =
    R"(#version 330 core

uniform vec3 camera_position;

uniform vec3 albedo;

uniform vec3 sun_direction;
uniform vec3 sun_color;

in vec3 position;
in vec3 normal;

layout (location = 0) out vec4 out_color;

vec3 diffuse(vec3 direction) {
    return albedo * max(0.0, dot(normal, direction));
}

vec3 specular(vec3 direction) {
    float power = 64.0;
    vec3 reflected_direction = 2.0 * normal * dot(normal, direction) - direction;
    vec3 view_direction = normalize(camera_position - position);
    return albedo * pow(max(0.0, dot(reflected_direction, view_direction)), power);
}

vec3 phong(vec3 direction) {
    return diffuse(direction) + specular(direction);
}

void main()
{
    float ambient_light = 0.2;
    vec3 color = albedo * ambient_light + sun_color * phong(sun_direction);
    out_color = vec4(color, 1.0);
}
)";
*/


/*
// === Задание 4 ===
const char fragment_shader_source[] =
    R"(#version 330 core

uniform sampler2D shadow_texture;  // текстура shadow map - она для отрисовки тени будет испольщоваться
uniform mat4 shadow_projection;  // и матрица проекции, с помощью которой можно взглянуть на мир так, как будто находимся в источнике света

uniform vec3 camera_position;
uniform vec3 albedo;
uniform vec3 sun_direction;
uniform vec3 sun_color;
in vec3 position;
in vec3 normal;

layout (location = 0) out vec4 out_color;

vec3 diffuse(vec3 direction) {
    return albedo * max(0.0, dot(normal, direction));
}

vec3 specular(vec3 direction) {
    float power = 64.0;
    vec3 reflected_direction = 2.0 * normal * dot(normal, direction) - direction;
    vec3 view_direction = normalize(camera_position - position);
    return albedo * pow(max(0.0, dot(reflected_direction, view_direction)), power);
}

vec3 phong(vec3 direction) {
    return diffuse(direction) + specular(direction);
}

void main()
{
    float ambient_light = 0.2;
    vec3 color = albedo * ambient_light;  // пока уитываем в цвете только ambient освещение

    vec4 ndc = shadow_projection * vec4(position, 1.0);  // координаты того, как выглядит точка с точки зрения источника света (из которого весь мир веден согласно матрице shadow_projection)
    if (abs(ndc.x) <= 1.0 && abs(ndc.y) <= 1.0) {  // координаты точки должны быть в видимой области (вообще так всегда автоматически получится, если мы нормально выбрали матрицу проекции, чтобы источник света видел всю сцену)
        vec2 shadow_texcoord = ndc.xy * 0.5 + vec2(0.5);  // так как координаты после проекции стали формата opengl (от -1 до 1 по всем осям), а при этом нумерация в текстуре происходит от 0 до 1 (и при этом мы точно знаем, что эти [-1, 1] opengl-сцены превратились в [0,1] текстуры биективно, так как перед рисование текстуры ставили glViewport(0, 0, shadow_map_size, shadow_map_size); -> весь экран [-1,1] замарлен на всю текстуру), то просто равномерно растягиваем их, переводя от 0 до 1
        float shadow_depth = ndc.z * 0.5 + 0.5;

        if (texture(shadow_texture, shadow_texcoord).r < shadow_depth) {  // если так, значит из источника света данная точка (пиксель который мы рисуем в данном шейдере) не виден - его перекрывает другой объект (с меньшим значением буфера глубины) -> для него не применяем прямое освещение от источника
            // без изменений
        } else {
            color += sun_color * phong(sun_direction);  // иначе добавляем прямое освещение
        }
    } else
        color += sun_color * phong(sun_direction);  // важно! видимо у нас всё же не очень матрица проекции, которая види не всю сцену (но главного персонажа видно, так что ок) -> для точек вне сцены оставляем прямое освещение, иначе темно будет на границе плоскости с фигуркой... !!! на самом деле, это потому, что я забыл вектора light_X и другие отнормировать!

    out_color = vec4(color, 1.0);
}
)";
*/


/*
// === Задание 6 ===
const char fragment_shader_source[] =
    R"(#version 330 core

uniform sampler2DShadow shadow_texture;  // заменяем на sampler2DShadow
uniform mat4 shadow_projection;

uniform vec3 camera_position;
uniform vec3 albedo;
uniform vec3 sun_direction;
uniform vec3 sun_color;
in vec3 position;
in vec3 normal;

layout (location = 0) out vec4 out_color;

vec3 diffuse(vec3 direction) {
    return albedo * max(0.0, dot(normal, direction));
}

vec3 specular(vec3 direction) {
    float power = 64.0;
    vec3 reflected_direction = 2.0 * normal * dot(normal, direction) - direction;
    vec3 view_direction = normalize(camera_position - position);
    return albedo * pow(max(0.0, dot(reflected_direction, view_direction)), power);
}

vec3 phong(vec3 direction) {
    return diffuse(direction) + specular(direction);
}

void main()
{
    float ambient_light = 0.2;
    vec3 color = albedo * ambient_light;

    vec4 ndc = shadow_projection * vec4(position, 1.0);
    if (abs(ndc.x) <= 1.0 && abs(ndc.y) <= 1.0) {
        vec2 shadow_texcoord = ndc.xy * 0.5 + vec2(0.5);
        float shadow_depth = ndc.z * 0.5 + 0.5;
        сolor += texture(shadow_texture, vec3(shadow_texcoord, shadow_depth)) * sun_color * phong(sun_direction);  // прибавляем цвет с коэффициентом = сила затемнения (он от 0 (=тень) до 1 (=свет))
    } else
        color += sun_color * phong(sun_direction);

    out_color = vec4(color, 1.0);
}
)";
*/


// ==== Задание 7 ====
const char fragment_shader_source[] =
    R"(#version 330 core

uniform sampler2DShadow shadow_texture; 
uniform mat4 shadow_projection;

uniform vec3 camera_position;
uniform vec3 albedo;
uniform vec3 sun_direction;
uniform vec3 sun_color;
in vec3 position;
in vec3 normal;

layout (location = 0) out vec4 out_color;

vec3 diffuse(vec3 direction) {
    return albedo * max(0.0, dot(normal, direction));
}

vec3 specular(vec3 direction) {
    float power = 64.0;
    vec3 reflected_direction = 2.0 * normal * dot(normal, direction) - direction;
    vec3 view_direction = normalize(camera_position - position);
    return albedo * pow(max(0.0, dot(reflected_direction, view_direction)), power);
}

vec3 phong(vec3 direction) {
    return diffuse(direction) + specular(direction);
}

void main()
{
    float ambient_light = 0.2;
    vec3 color = albedo * ambient_light;

    vec4 ndc = shadow_projection * vec4(position, 1.0);  // снова берём координаты точки (соответсвтующей пикселю, для которого вызвался шейдер), как если смотреть на неё из источника света (с направления Солнца)
    if (!(abs(ndc.x) <= 1.0 && abs(ndc.y) <= 1.0)) {  // если точка не попала на сцену, значит ситаем, что она и так освещена (просто значит она не в центре сцены, значит не особо интересна -> пусть освещена; на практике это край плоскости, на которой стоит фиугрка)
        color += sun_color * phong(sun_direction);
        out_color = vec4(color, 1.0);
        return;
    }

    vec2 shadow_texcoord = ndc.xy * 0.5 + vec2(0.5);  // снова получаем координаты в текстуре и истинное значение глубины рассматриваемой точки
    float shadow_depth = ndc.z * 0.5 + 0.5;

    // Гауссово размытие - адаптированный код из лекции 6;
    // мы должны размыть как раз коэффициент texture(shadow_texture, vec3(shadow_texcoord, shadow_depth)), на которой домножить добавку света от Солнца - аналогично предыдущему коду (к заданию 6), но там не размывалм
    float sum = 0.0;  // здесь суммируем значения этих коэффициентов
    float sum_w = 0.0;  // а здесь вес, на которой нужно будет поделить
    const int N = 7;
    float radius = 5.0;
    for (int x = -N; x <= N; ++x) {
        for (int y = -N; y <= N; ++y) {
            vec2 offset = vec2(x,y) / vec2(textureSize(shadow_texture, 0));  // сдвиг на (x,y) от рассматриваемой точки соответсвует сдвигу offset на текстуре (помним, что там от 0 до 1 координаты -> нужно вот так нормировать, деля на размер текстуры) 
            float c = exp(-float(x*x + y*y) / (radius*radius));
            //shadow_depth = (shadow_projection * vec4(position + vec3(x,y,0.0), 1.0)).z * 0.5 + 0.5; - казалось бы, можно ещё обновить ожидаемое значение глубины у точки, сдвинувшись x,y - но так вроде не надо делать!  ТАК НЕЛЬЗЯ, ведь сдвиг (x,y) должен происходить уже после применения shadow_projection (но в этом случае смысла нет менять глуину, так как это сдвиг в плоскости, перпендикулярной направлению на источник -> у всех точек тут глубина одинакова)
            sum += c * texture(shadow_texture, vec3(shadow_texcoord + offset, shadow_depth));  // добавляем значение коэффициента на данном offset
            sum_w += c;
        }
    }
    color += (sum / sum_w) * sun_color * phong(sun_direction);  // прибавляем свет от Солнца с размытым коэфициентом
    out_color = vec4(color, 1.0);
}
)";



// ==== Задание 2 ====
const char debug_vertex_shader_source[] =
    R"(#version 330 core

    const vec2 VERTICES[6] = vec2[6](  // захардкоженные вершины
        vec2(-1.0, -0.5),
        vec2(-1.0, -1.0),
        vec2(-0.5, -0.5),
        vec2(-1.0, -1.0),
        vec2(-0.5, -1.0),
        vec2(-0.5, -0.5)
    );

    out vec2 texcoord;

void main()
{
    gl_Position = vec4(VERTICES[gl_VertexID], 0.0, 1.0);  // просто через индекс (который дается для каждого вызова шейдера - а вызывать мы его как раз будет 6 раз, для каждой из точек) достём коордианты вершины
    texcoord = VERTICES[gl_VertexID] * 2 + vec2(2, 2);  // сдвигаем и экстраполируем координаты для текстуры, чтобы они были от 0 до 1 -> покрывали всю текстуру
}
)";

const char debug_fragment_shader_source[] =
    R"(#version 330 core

uniform sampler2D sampler;

in vec2 texcoord;  // координаты в текстуре, которые пришли из вершинного шейдера (точнее, их интерполированные для каждого пикселя варианты)

layout (location = 0) out vec4 out_color;

void main()
{
    out_color = vec4(texture(sampler, texcoord).r);  // просто берем по данным координатам из текстуры коасный цвет (первая компонента через .r) - и дуюлируем его на все координаты (получается r = g = b -> серый цвет)
}
)";



// === Задание 3 ===
const char shadow_vertex_shader_source[] = R"(#version 330 core
layout (location = 0) in vec3 in_position;

uniform mat4 model;
uniform mat4 shadow_projection;

void main()
{
    gl_Position = shadow_projection * model * vec4(in_position, 1.0);  // in_position - координаты в модели, их умножаем на model (переводим в мировые коордианты), а затем на матрицу проекции (получаем проекцию, видимую из той точки, для котрой матрица) - должны уже получить координаты от -1 до 1 для opengl 
                                                                       // (обычно это делают view и projection вместе - и проекция, и перевод в нужные координаты, но тут странно выделять view, так как никакой камеры нет, поэтому прото projection)
}
)";

const char shadow_fragment_shader_source[] = R"(#version 330 core
void main() {}  // Фрагментный шейдер ничего не делает (пустая функция main; глубина пикселя, которая нам и нужна, пишется сама, автоматически)
)";




GLuint create_shader(GLenum type, const char *source)
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

int main()
try
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

    SDL_Window *window = SDL_CreateWindow("Graphics course practice 8",
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
    GLuint sun_direction_location = glGetUniformLocation(program, "sun_direction");
    GLuint sun_color_location = glGetUniformLocation(program, "sun_color");

    std::string project_root = PROJECT_ROOT;
    std::string scene_path = project_root + "/buddha.obj";
    obj_data scene = parse_obj(scene_path);

    GLuint scene_vao, scene_vbo, scene_ebo;
    glGenVertexArrays(1, &scene_vao);
    glBindVertexArray(scene_vao);

    glGenBuffers(1, &scene_vbo);
    glBindBuffer(GL_ARRAY_BUFFER, scene_vbo);
    glBufferData(GL_ARRAY_BUFFER, scene.vertices.size() * sizeof(scene.vertices[0]), scene.vertices.data(), GL_STATIC_DRAW);

    glGenBuffers(1, &scene_ebo);
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, scene_ebo);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, scene.indices.size() * sizeof(scene.indices[0]), scene.indices.data(), GL_STATIC_DRAW);

    glEnableVertexAttribArray(0);
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, sizeof(obj_data::vertex), (void *)(0));
    glEnableVertexAttribArray(1);
    glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, sizeof(obj_data::vertex), (void *)(12));

    auto last_frame_start = std::chrono::high_resolution_clock::now();

    float time = 0.f;

    std::map<SDL_Keycode, bool> button_down;

    float camera_distance = 1.5f;
    float camera_angle = glm::pi<float>();




    // ========= МОЙ КОД: ============


    // === Задание 1 ===
    int shadow_map_size = 1024;
    GLuint shadow_map_texture;
    glGenTextures(1, &shadow_map_texture);  // создали объект текстуры для shadow map
    glActiveTexture(GL_TEXTURE0 + 0);  // как обычно, активируем texture unit 0 (он и так по умолчанию активен, но все же)
    glBindTexture(GL_TEXTURE_2D, shadow_map_texture);  // и только в texture unit можем делать текущей текстуру
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_DEPTH_COMPONENT24, shadow_map_size, shadow_map_size,
                 0, GL_DEPTH_COMPONENT, GL_FLOAT, nullptr);  // задали размер и цветовой формат
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);

    GLuint framebuf;
    glGenFramebuffers(1, &framebuf);  // создаём и делаем текущим frame buffer (если после этого ничего не менять, то будет черны экран, тк рисовать будем в данный буфер, а не в дефольный, который на экране)
    glBindFramebuffer(GL_DRAW_FRAMEBUFFER, framebuf);                                                  
    glFramebufferTexture(GL_DRAW_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, shadow_map_texture, 0);  // прикрепляем 0-ой уровень (а мы только его и имеем, mipmap мы не активировали и не генерировали) нашей текстуры как буфер
    if (glCheckFramebufferStatus(GL_DRAW_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE) {
        perror("Frame buffer некорректно настроен!");
        exit(1);
    }



    // === Задание 6 ===
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_COMPARE_MODE, GL_COMPARE_REF_TO_TEXTURE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_COMPARE_FUNC,  GL_LEQUAL);
    // кстати, в задании сказано, что теперь не будет работать прямоугольник внизу экрана - но у меня, например, работает 
    // (на самом деле - как повезёт, но по спецификации работать не должен...)



    // === Задание 2 ===
    auto debug_vertex_shader = create_shader(GL_VERTEX_SHADER, debug_vertex_shader_source);  // компилируем программу шейдерную для дебага, чтобы рисовать прямоугольник из захардкоденных вершин
    auto debug_fragment_shader = create_shader(GL_FRAGMENT_SHADER, debug_fragment_shader_source);
    auto debug_program = create_program(debug_vertex_shader, debug_fragment_shader);

    GLuint debug_vao;
    glGenVertexArrays(1, &debug_vao);  // фиктивный vao (он всегда нужен для рисования, но в данном случае он пустой, тк все вершины и так захардкожены)
    


    // === Задание 3 ===
    auto shadow_vertex_shader = create_shader(GL_VERTEX_SHADER, shadow_vertex_shader_source);  // компилируем очередную программу
    auto shadow_fragment_shader = create_shader(GL_FRAGMENT_SHADER, shadow_fragment_shader_source);
    auto shadow_program = create_program(shadow_vertex_shader, shadow_fragment_shader);
    GLuint shadow_model_location = glGetUniformLocation(shadow_program, "model");  // получаем положение переменных
    GLuint shadow_projection_location = glGetUniformLocation(shadow_program, "shadow_projection");



    // === Задание 4 ===
    GLuint program_shadow_projection_location = glGetUniformLocation(program, "shadow_projection");  // находим положение матрицы проекции в основной программе



    bool running = true;
    while (running)
    {
        for (SDL_Event event; SDL_PollEvent(&event);)
            switch (event.type)
            {
            case SDL_QUIT:
                running = false;
                break;
            case SDL_WINDOWEVENT:
                switch (event.window.event)
                {
                case SDL_WINDOWEVENT_RESIZED:
                    width = event.window.data1;
                    height = event.window.data2;
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
            camera_angle += 2.f * dt;
        if (button_down[SDLK_RIGHT])
            camera_angle -= 2.f * dt;
        


        /*
        // === Задание 3 === - рисуем копию основной сцены в shadow mapping текстуру, которая затем прямоугольником внизу экрана отобразится
        glBindFramebuffer(GL_DRAW_FRAMEBUFFER, framebuf);  // переключаемся на наш фреймбуфер (где текстура для shadow mapping) - в неё сейчас будем рисовать
        glViewport(0, 0, shadow_map_size, shadow_map_size);
        glClear(GL_DEPTH_BUFFER_BIT);  // очистили бкфер глубины и настроили все остальное
        glClearColor(0.8f, 0.8f, 1.f, 0.f);
        glEnable(GL_DEPTH_TEST);
        glEnable(GL_CULL_FACE);  // зачем ? - чтобы избежать артефакта как обсуждаи на лекции
        glCullFace(GL_FRONT);

        glm::vec3 light_Z = glm::vec3(0, -1, 0);
        glm::vec3 light_X = glm::vec3(1, 0, 0);
        glm::vec3 light_Y = glm::cross(light_X, light_Z);
        glm::mat4 shadow_projection = glm::mat4(glm::transpose(glm::mat3(light_X, light_Y, light_Z)));  // настроили матрицу как описано
        glm::mat4 shadow_model(1.f);  // матрицу модели (перводящую координаты модели в мировые) берём такую же, как ниже для рисования основной сцены
        
        glUseProgram(shadow_program);  // переключились на программу и в рамках неё настроили все переменные
        glUniformMatrix4fv(shadow_model_location, 1, GL_FALSE, reinterpret_cast<float *>(&shadow_model));
        glUniformMatrix4fv(shadow_projection_location, 1, GL_FALSE, reinterpret_cast<float *>(&shadow_projection));
        glBindTexture(GL_TEXTURE_2D, shadow_map_texture);
        glBindVertexArray(scene_vao);  // берем тот же vao, что для основной сцены и рисуем её в наш фрембуфер:
        glDrawElements(GL_TRIANGLES, scene.indices.size(), GL_UNSIGNED_INT, nullptr);  // теперь текстура (использовавшаяся как буфер глубины) заполнилась и будет отрисовываться внизу экрана

        glCullFace(GL_BACK);  // не забываем вернуть обратно back-face culling и далее уже пойдут настройки для рисования основной сцены на экран
        */

        // === Задание 5 ===
        glBindFramebuffer(GL_DRAW_FRAMEBUFFER, framebuf);
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
        glViewport(0, 0, shadow_map_size, shadow_map_size);
        glClearColor(0.8f, 0.8f, 1.f, 0.f);
        glEnable(GL_DEPTH_TEST);
        glEnable(GL_CULL_FACE);
        glCullFace(GL_FRONT);

        glm::vec3 light_direction = glm::normalize(glm::vec3(std::sin(time * 0.5f), 2.f, std::cos(time * 0.5f)));  // та же функция, что и ниже (при рисовании основной сцены) = направление на солнце (ведь хотим тень от солнца)
        glm::vec3 light_Z = -light_direction;
        glm::vec3 light_X = glm::normalize(glm::cross(light_Z, {0.f, 1.f, 0.f}));  // векторное произведение -> какой-то ортогональный light_z вектор
        glm::vec3 light_Y = glm::normalize(glm::cross(light_X, light_Z));  // обязательно нормируем все векторы!!!! иначе могут быть артефакты (в предыдущем закомментированном коде, кстати, к заданию 3 не нормировал - но там ничего страшного, кроме того, что сцена не влезала в shadow_map -> пришлось писать строчку 161)
        glm::mat4 shadow_projection = glm::mat4(glm::transpose(glm::mat3(light_X, light_Y, light_Z)));
        glm::mat4 shadow_model(1.f);
        
        glUseProgram(shadow_program);
        glUniformMatrix4fv(shadow_model_location, 1, GL_FALSE, reinterpret_cast<float *>(&shadow_model));
        glUniformMatrix4fv(shadow_projection_location, 1, GL_FALSE, reinterpret_cast<float *>(&shadow_projection));
        glBindTexture(GL_TEXTURE_2D, shadow_map_texture);
        glBindVertexArray(scene_vao);
        glDrawElements(GL_TRIANGLES, scene.indices.size(), GL_UNSIGNED_INT, nullptr);
        
        glCullFace(GL_BACK);




        // === Задание 2 ===
        glBindFramebuffer(GL_DRAW_FRAMEBUFFER, 0);  // делаем дефолтный фреймбуер снова текущим, чтобы рисовать на экран

        
        // === Задание 4 ===
        glUseProgram(program); 
        glUniformMatrix4fv(program_shadow_projection_location, 1, GL_FALSE, reinterpret_cast<float *>(&shadow_projection));  // передаём ту же матрицу проекции, с помощью которой видим сцену из источника


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
        view = glm::rotate(view, glm::pi<float>() / 6.f, {1.f, 0.f, 0.f});
        view = glm::rotate(view, camera_angle, {0.f, 1.f, 0.f});
        view = glm::translate(view, {0.f, -0.5f, 0.f});

        float aspect = (float)height / (float)width;
        glm::mat4 projection = glm::perspective(glm::pi<float>() / 3.f, (width * 1.f) / height, near, far);

        glm::vec3 camera_position = (glm::inverse(view) * glm::vec4(0.f, 0.f, 0.f, 1.f)).xyz();

        glm::vec3 sun_direction = glm::normalize(glm::vec3(std::sin(time * 0.5f), 2.f, std::cos(time * 0.5f)));

        glUseProgram(program);

        glUniformMatrix4fv(model_location, 1, GL_FALSE, reinterpret_cast<float *>(&model));
        glUniformMatrix4fv(view_location, 1, GL_FALSE, reinterpret_cast<float *>(&view));
        glUniformMatrix4fv(projection_location, 1, GL_FALSE, reinterpret_cast<float *>(&projection));
        glUniform3fv(camera_position_location, 1, (float *)(&camera_position));
        glUniform3f(albedo_location, .8f, .7f, .6f);
        glUniform3f(sun_color_location, 1.f, 1.f, 1.f);
        glUniform3fv(sun_direction_location, 1, reinterpret_cast<float *>(&sun_direction));

        glBindVertexArray(scene_vao);
        glDrawElements(GL_TRIANGLES, scene.indices.size(), GL_UNSIGNED_INT, nullptr);


        // === Задание 2 ===
        glUseProgram(debug_program);  // переключаемся на нужную программу
        glBindTexture(GL_TEXTURE_2D, shadow_map_texture);  // делаем текущей текстуру
        glDisable(GL_DEPTH_TEST);  // выключаем тест глубины, чтобы прямоугольник не оказался ‘за’ основной сценой
        glBindVertexArray(debug_vao);  // делаем текущим vao, по которому и будет всё рисоваться
        glDrawArrays(GL_TRIANGLES, 0, 6);  // рисуем 6 точек (их индекс будет от 0 до 5 -> как нам и нужно в шейдере)
        // По-хорошему для связи sampler2D и текстуры нужен texture unit; для простоты можем воспользоваться тем, что shadow_map_texture была у нас в unit = 0, и значение uniform-переменных по умолчанию – тоже ноль


        SDL_GL_SwapWindow(window);
    }

    SDL_GL_DeleteContext(gl_context);
    SDL_DestroyWindow(window);
}
catch (std::exception const &e)
{
    std::cerr << e.what() << std::endl;
    return EXIT_FAILURE;
}
