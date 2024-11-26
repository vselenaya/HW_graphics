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
#include <random>
#include <map>
#include <cmath>

#define GLM_FORCE_SWIZZLE
#define GLM_ENABLE_EXPERIMENTAL
#include <glm/vec3.hpp>
#include <glm/mat4x4.hpp>
#include <glm/ext/matrix_transform.hpp>
#include <glm/ext/matrix_clip_space.hpp>
#include <glm/ext/scalar_constants.hpp>
#include <glm/gtx/quaternion.hpp>
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

/* было изначально
const char vertex_shader_source[] =
R"(#version 330 core

layout (location = 0) in vec3 in_position;

void main()
{
    gl_Position = vec4(in_position, 1.0);
}
)";

const char geometry_shader_source[] =
R"(#version 330 core

uniform mat4 model;
uniform mat4 view;
uniform mat4 projection;
uniform vec3 camera_position;

layout (points) in;
layout (points, max_vertices = 1) out;

void main()
{
    vec3 center = gl_in[0].gl_Position.xyz;
    gl_Position = projection * view * model * vec4(center, 1.0);
    EmitVertex();
    EndPrimitive();
}

)";
*/

/*
// === Задание 1 ===
const char vertex_shader_source[] =
R"(#version 330 core

layout (location = 0) in vec3 in_position;
layout (location = 1) in float in_size;  // добавили атрибут вершины - размер
out float size;  // а также добавляем возвращаемое значение,  которое полетит в геом шейдер

void main()
{
    gl_Position = vec4(in_position, 1.0);
    size = in_size;
}
)";
*/

// === Задание 5 ===
const char vertex_shader_source[] =
R"(#version 330 core

layout (location = 0) in vec3 in_position;
layout (location = 1) in float in_size;
layout(location = 2) in float in_rotation; // Угол поворота

out float size;
out float rotation;

void main()
{
    gl_Position = vec4(in_position, 1.0);
    size = in_size;
    rotation = in_rotation;  // передаем угол поворота дальше - в геометрический шейдер
}
)";

/* было в задании 1
const char geometry_shader_source[] =
R"(#version 330 core

uniform mat4 model;
uniform mat4 view;
uniform mat4 projection;
uniform vec3 camera_position;

layout (points) in;
layout (triangle_strip, max_vertices = 4) out;  // на выходе получаем 4 вершины, которые образую прямоугольник из двух треугольников
in float size[];  // size, который пришел из вершинного шейдера (геометрический шейдер обрабатывает сразу целый примитив, т.е. несколько вершин, поэтому нужен массив... в нашем случае рисуются точки (так вызываем функцию рисования с используемыми шейдерами), поэтомуданный массив состоит из одного элемента)

void main()
{
    vec3 center = gl_in[0].gl_Position.xyz;

    // Генерация 4 вершин для прямоугольника (после каждой вызывается EmitVertex(), добавляя их в выходной поток)
    gl_Position = projection * view * model * vec4(center + vec3(-size[0], -size[0], 0.0), 1.0);
    EmitVertex();

    gl_Position = projection * view * model * vec4(center + vec3(size[0], -size[0], 0.0), 1.0);
    EmitVertex();

    gl_Position = projection * view * model * vec4(center + vec3(-size[0], size[0], 0.0), 1.0);
    EmitVertex();

    gl_Position = projection * view * model * vec4(center + vec3(size[0], size[0], 0.0), 1.0);
    EmitVertex();

    EndPrimitive();  // после всех 4 вершин наш примитив окончен -> все 4 вершины можно превратить в triangle-strip, то есть в прямоугольник
}
)";
*/

/*
// === Задание 2 ===
const char geometry_shader_source[] =
R"(#version 330 core

uniform mat4 model;
uniform mat4 view;
uniform mat4 projection;
uniform vec3 camera_position;

layout (points) in;
layout (triangle_strip, max_vertices = 4) out;
in float size[];
out vec2 texcoord; // Выходные текстурные координаты (они пойдут во фрагментный шейдер)

void main()
{
    vec3 center = gl_in[0].gl_Position.xyz;
    gl_Position = projection * view * model * vec4(center + vec3(-size[0], -size[0], 0.0), 1.0);
    texcoord = vec2(0.0, 0.0); // Нижний левый угол
    EmitVertex();
    gl_Position = projection * view * model * vec4(center + vec3(size[0], -size[0], 0.0), 1.0);
    texcoord = vec2(1.0, 0.0); // Нижний правый угол
    EmitVertex();
    gl_Position = projection * view * model * vec4(center + vec3(-size[0], size[0], 0.0), 1.0);
    texcoord = vec2(0.0, 1.0); // Верхний левый угол
    EmitVertex();
    gl_Position = projection * view * model * vec4(center + vec3(size[0], size[0], 0.0), 1.0);
    texcoord = vec2(1.0, 1.0); // Верхний правый угол
    EmitVertex();
    EndPrimitive();
}
)";
*/

/*
// === Задание 3 ===
const char geometry_shader_source[] =
R"(#version 330 core

uniform mat4 model;
uniform mat4 view;
uniform mat4 projection;
uniform vec3 camera_position;

layout (points) in;
layout (triangle_strip, max_vertices = 4) out;
in float size[];
out vec2 texcoord;

void main()
{
    vec3 center = gl_in[0].gl_Position.xyz;
    vec3 Z = normalize(camera_position - center);  // Вычисляем направление Z от частицы (её центра) к камере

    // Вычисляем произвольный вектор X, перпендикулярный Z; для этого сначала берем любой вектор, который не совпадает с Z:
    vec3 arbitrary_vector = vec3(0.0, 1.0, 0.0);
    if (abs(Z.y) > 0.99)
        arbitrary_vector = vec3(1.0, 0.0, 0.0); // Если Z близок к вертикали, используем другой вектор (чтобы точно с Z не совпадало)
    vec3 X = normalize(cross(arbitrary_vector, Z)); // Перпендикулярный вектор X
    vec3 Y = cross(Z, X); // Перпендикулярный вектор Y (итого X, Y, Z - ортонормированный (те взаимоперепендикулярны и длины 1 (тк normalize) векторы) базис)

    gl_Position = projection * view * model * vec4(center + X * size[0] + Y * size[0], 1.0);  // чтобы частица была в плоскости XY, просто рисуем её прямоугольник +-size[0] в плоскости XY -> те по координатм X и Y делаем +-size[0]
    texcoord = vec2(0.0, 0.0);
    EmitVertex();
    gl_Position = projection * view * model * vec4(center + X * size[0] - Y * size[0], 1.0);
    texcoord = vec2(1.0, 0.0);
    EmitVertex();
    gl_Position = projection * view * model * vec4(center - X * size[0] + Y * size[0], 1.0);
    texcoord = vec2(0.0, 1.0);
    EmitVertex();
    gl_Position = projection * view * model * vec4(center - X * size[0] - Y * size[0], 1.0);
    texcoord = vec2(1.0, 1.0);
    EmitVertex();
    EndPrimitive();
}
)";
*/


// === Задание 5 ===
const char geometry_shader_source[] =
R"(#version 330 core

uniform mat4 model;
uniform mat4 view;
uniform mat4 projection;
uniform vec3 camera_position;

layout (points) in;
layout (triangle_strip, max_vertices = 4) out;
in float size[];
in float rotation[]; // Угол поворота из вершинного шейдера
out vec2 texcoord;

void main()
{
    vec3 center = gl_in[0].gl_Position.xyz;

    // оси:
    vec3 Z = normalize(camera_position - center);  
    vec3 arbitrary_vector = vec3(0.0, 1.0, 0.0);
    if (abs(Z.y) > 0.99)
        arbitrary_vector = vec3(1.0, 0.0, 0.0);
    vec3 X = normalize(cross(arbitrary_vector, Z));
    vec3 Y = cross(Z, X);

    // Создаем матрицу смены базиса - она переводит вектор:
    // -> из базиса, составленного из полученных нами векторов X, Y, Z
    // -> в стандартный ортонормированного базиса x,y,z (каждый вектор = столбец из нулей с одной единичкой: x = (1,0,0), y=(0,1,0), z=(0,0,1))
    mat3 change_of_basis = mat3(X, Y, Z);  // матрица, где столбцы = векторы X, Y, Z (помним, что в opengl матрицы транспонированно задаются)
                                           // самопроверка: вектор (1,0,0) в нашем новом базисе (из X, Y, Z) перейдет в вектор X в стандартном базисе (логично, тк (1,0,0) это обозначения x-вектора в текщуем базисе)

    // Создаем матрицу поворота вокруг оси z (в любом базисе)
    float cos_angle = cos(rotation[0]);
    float sin_angle = sin(rotation[0]);
    mat3 rotation_matrix = mat3(
        cos_angle, -sin_angle, 0.0,
        sin_angle,  cos_angle, 0.0,
        0.0,        0.0,       1.0
    );

    // Поворачиваем X и Y с помощью матрицы поворота
    // тк мы хотим делать поворот в плоскости XY (= вокруг оси Z), мы сначала меняем базис вектора на наш (X, Y, Z) - там ось z равна вектору Z ->
    // -> затем делаем попорот вокруг z -> затем возвращаем исходный базис
    vec3 rotated_X = change_of_basis * rotation_matrix * inverse(change_of_basis) * X;
    vec3 rotated_Y = change_of_basis * rotation_matrix * inverse(change_of_basis) * Y;

    // rotated_X = cos(angle) * X + sin * Y; - можно было так сделать (попроще)... это просто двумерная матрица поворота, которая сразу в плоскости XY работает
    // rotated_Y = -sin * X + cos * Y;

    // Генерация 4 вершин для прямоугольника, параллельного плоскости XY (как и раньше, но повернуты X, Y)
    gl_Position = projection * view * model * vec4(center + rotated_X * size[0] + rotated_Y * size[0], 1.0);
    texcoord = vec2(0.0, 0.0);
    EmitVertex();
    gl_Position = projection * view * model * vec4(center + rotated_X * size[0] - rotated_Y * size[0], 1.0);
    texcoord = vec2(1.0, 0.0);
    EmitVertex();
    gl_Position = projection * view * model * vec4(center - rotated_X * size[0] + rotated_Y * size[0], 1.0);
    texcoord = vec2(0.0, 1.0);
    EmitVertex();
    gl_Position = projection * view * model * vec4(center - rotated_X * size[0] - rotated_Y * size[0], 1.0);
    texcoord = vec2(1.0, 1.0);
    EmitVertex();
    EndPrimitive();
}
)";


/* было изначально
const char fragment_shader_source[] =
R"(#version 330 core

layout (location = 0) out vec4 out_color;

void main()
{
    out_color = vec4(1.0, 0.0, 0.0, 1.0);
}
)";
*/

/*
// === Задание 2 ===
const char fragment_shader_source[] =
R"(#version 330 core

in vec2 texcoord; // Входные текстурные координаты (они интеполируются между вершинами прямоугольника, рисуемог в геометрическом шейдере)
layout (location = 0) out vec4 out_color;

void main()
{
    out_color = vec4(texcoord, 0.0, 1.0);
}
)";
*/

/*
// === Задание 7 ===
const char fragment_shader_source[] =
R"(#version 330 core

in vec2 texcoord;
layout (location = 0) out vec4 out_color;
uniform sampler2D particleTexture; // Текстура частиц

void main()
{
    vec4 texColor = texture(particleTexture, texcoord);  // Получаем цвет из текстуры
    float alpha = texColor.r; // Используем красный (первый) канал как альфа
    out_color = vec4(1.0, 1.0, 1.0, alpha); // Белый цвет с альфа-каналом
}
)";
*/


// === Задание 8 ===
const char fragment_shader_source[] =
R"(#version 330 core

in vec2 texcoord;
layout (location = 0) out vec4 out_color;
uniform sampler2D particleTexture;
uniform sampler1D colorPalette; // Текстура палитры

void main()
{
    float alpha = texture(particleTexture, texcoord).r;  // альфа канал из текстуры
    vec4 paletteColor = texture(colorPalette, alpha);  // им как индекосм получаем цвет
    out_color = vec4(paletteColor.rgb, alpha);  // собираем итоговый цвет как цвет + альфа канал
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


/* было изначально
struct particle
{
    glm::vec3 position;
};
*/

/*
// === Задание 1 ===
struct particle
{
    glm::vec3 position;
    float size;
};
*/

/*
// === Задание 4 ===
struct particle
{
    glm::vec3 position;
    float size;
    glm::vec3 velocity;
};
*/

// === Задание 6 ===
std::default_random_engine part_rng;  // создаём рандомное устройство

// === Задание 5 ===
struct particle
{
    glm::vec3 position;
    float size;
    glm::vec3 velocity;
    float rotation; // Угол поворота
    float angular_velocity; // Угловая скорость

    // === Задание 6 ===
    particle() {  // пишем конструктор, чтобы легко пересоздавать, не инициалиируя вручную...
        position.x = std::uniform_real_distribution<float>{-1.f, 1.f}(part_rng);
        position.y = 0.f;
        position.z = std::uniform_real_distribution<float>{-1.f, 1.f}(part_rng);
        size = std::uniform_real_distribution<float>{0.2f, 0.4f}(part_rng);
        velocity.x = std::uniform_real_distribution<float>{0.1, 0.2}(part_rng);
        velocity.y = std::uniform_real_distribution<float>{0.1, 0.2}(part_rng);
        velocity.z = std::uniform_real_distribution<float>{0.1, 0.2}(part_rng);
        rotation = 0.0f;
        angular_velocity = std::uniform_real_distribution<float>{0.f, 2.f * M_PI}(part_rng);
    }
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

    SDL_Window * window = SDL_CreateWindow("Graphics course practice 11",
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

    glClearColor(0.f, 0.f, 0.f, 0.f);

    auto vertex_shader = create_shader(GL_VERTEX_SHADER, vertex_shader_source);
    auto geometry_shader = create_shader(GL_GEOMETRY_SHADER, geometry_shader_source);
    auto fragment_shader = create_shader(GL_FRAGMENT_SHADER, fragment_shader_source);
    auto program = create_program(vertex_shader, geometry_shader, fragment_shader);

    GLuint model_location = glGetUniformLocation(program, "model");
    GLuint view_location = glGetUniformLocation(program, "view");
    GLuint projection_location = glGetUniformLocation(program, "projection");
    GLuint camera_position_location = glGetUniformLocation(program, "camera_position");

    std::default_random_engine rng;

    std::vector<particle> particles(256);
    for (auto & p : particles)
    {
        p.position.x = std::uniform_real_distribution<float>{-1.f, 1.f}(rng);
        p.position.y = 0.f;
        p.position.z = std::uniform_real_distribution<float>{-1.f, 1.f}(rng);
        
        // === Задание 1 ===
        p.size = std::uniform_real_distribution<float>{0.2f, 0.4f}(rng);  // случайный размер

        // === Задание 4 ===
        p.velocity.x = std::uniform_real_distribution<float>{0.1, 0.2}(rng);
        p.velocity.y = std::uniform_real_distribution<float>{0.1, 0.2}(rng);
        p.velocity.z = std::uniform_real_distribution<float>{0.1, 0.2}(rng);

        // === Задание 5 ===
        p.rotation = 0.0f; // Начальный угол поворота
        p.angular_velocity = std::uniform_real_distribution<float>{0.f, 2.f * M_PI}(rng); // Случайная угловая скорость от 0 до 2π
    }

    GLuint vao, vbo;
    glGenVertexArrays(1, &vao);
    glBindVertexArray(vao);

    glGenBuffers(1, &vbo);
    glBindBuffer(GL_ARRAY_BUFFER, vbo);

    glEnableVertexAttribArray(0);
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, sizeof(particle), (void*)(0));

    // === Задание 1 ===
    glEnableVertexAttribArray(1); // Индекс 1 аттрибута для размера size
    glVertexAttribPointer(1, 1, GL_FLOAT, GL_FALSE, sizeof(particle), (void*)(sizeof(glm::vec3))); 

    // === Задание 5 ===
    glEnableVertexAttribArray(2); // Индекс 2 аттрибута для угла поворота -> настраиваем:
    glVertexAttribPointer(2, 1, GL_FLOAT, GL_FALSE, sizeof(particle), (void*)(sizeof(glm::vec3)+sizeof(float)+sizeof(glm::vec3)));  // угол поворота = одно чсило float, смещение внутри particle равно vec3 (у position), затем float, затем еще раз vec3 (скорость)

    const std::string project_root = PROJECT_ROOT;
    const std::string particle_texture_path = project_root + "/particle.png";

    glPointSize(5.f);

    auto last_frame_start = std::chrono::high_resolution_clock::now();

    float time = 0.f;

    std::map<SDL_Keycode, bool> button_down;

    float view_angle = 0.f;
    float camera_distance = 2.f;
    float camera_height = 0.5f;

    float camera_rotation = 0.f;

    bool paused = false;



    // === Задание 7 ===
    // Загрузка текстуры как картинки из файла
    GLuint texture;
    int texture_width, texture_height, channels;
    unsigned char* data = stbi_load(particle_texture_path.data(), &texture_width, &texture_height, &channels, 4);  // требуемое число каналов = 4 (мы хотим, чтобы каждвй цвет был vec4 - то есть r, g, b, a... несмотря на то, что картинка в оттенках серого, оставшиеся каналы нулями дополнятся)
    if (!data)
        std::cerr << "Failed to load texture: " << std::endl;
    
    // Создаём текстуру:
    glGenTextures(1, &texture);
    glBindTexture(GL_TEXTURE_2D, texture);
    
    // Устанавливаем параметры текстуры и закгружаем в неё данные картинки
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, texture_width, texture_height, 0, GL_RGBA, GL_UNSIGNED_BYTE, data);
    stbi_image_free(data); // Освобождаем память

    // Настройка фильтрации
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glGenerateMipmap(GL_TEXTURE_2D);

    GLuint particleTexture_location = glGetUniformLocation(program, "particleTexture");  // получаем положение переменной = texture unit, в которой текущей будет сделана наша текстура (ну как и раньше передаем текстуру в шейдер)
    glUseProgram(program);
    glUniform1i(particleTexture_location, 0);  // передаем 0 - то есть наша текстура будет текущей в 0-м (по умолчанию активен) texture unit



    // === Задание 6 ===
    particles.clear();  // этот массив нам больше не нужен -> будем создавать вершины по одной



    // === Задание 8 ===
    // Цветовая палитра
    GLfloat palette[] = {
        0.0f, 0.0f, 0.0f, 1.0f, // Черный
        1.0f, 0.5f, 0.0f, 1.0f, // Оранжевый
        1.0f, 1.0f, 0.0f, 1.0f, // Желтый
        1.0f, 1.0f, 1.0f, 1.0f  // Белый
    };

    // Создание текстуры
    GLuint colorPaletteTexture;
    glGenTextures(1, &colorPaletteTexture);
    glBindTexture(GL_TEXTURE_1D, colorPaletteTexture);
    glTexImage1D(GL_TEXTURE_1D, 0, GL_RGBA, 4, 0, GL_RGBA, GL_FLOAT, palette);

    // Настройка фильтрации
    glTexParameteri(GL_TEXTURE_1D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_1D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);

    GLuint colorPalette_location = glGetUniformLocation(program, "colorPalette");
    glUseProgram(program);
    glUniform1i(colorPalette_location, 1);  // передаем 1 аналогично предыд заданию - то есть текстура палитры будет через 1-ый texture unit передаваться



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
            if (event.key.keysym.sym == SDLK_SPACE)
                paused = !paused;
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
            camera_distance -= 3.f * dt;
        if (button_down[SDLK_DOWN])
            camera_distance += 3.f * dt;

        if (button_down[SDLK_LEFT])
            camera_rotation -= 3.f * dt;
        if (button_down[SDLK_RIGHT])
            camera_rotation += 3.f * dt;

        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
        glEnable(GL_DEPTH_TEST);

        float near = 0.1f;
        float far = 100.f;

        glm::mat4 model(1.f);

        glm::mat4 view(1.f);
        view = glm::translate(view, {0.f, -camera_height, -camera_distance});
        view = glm::rotate(view, view_angle, {1.f, 0.f, 0.f});
        view = glm::rotate(view, camera_rotation, {0.f, 1.f, 0.f});

        glm::mat4 projection = glm::perspective(glm::pi<float>() / 2.f, (1.f * width) / height, near, far);

        glm::vec3 camera_position = (glm::inverse(view) * glm::vec4(0.f, 0.f, 0.f, 1.f)).xyz();


        // === Задание 6 ===
        if (!paused) {
            if (particles.size() < 256)
                particles.push_back(particle());  // добавляем новую частицу (ее конструктор, который добавли сам поля рандомом проинциализирует) пока их < 256
            for (auto &p: particles)
                if (p.position.y > 2.8)  // если у частицы y координата стала слишком большой (частица огня улетела слишком вверх), то ее как бы удаляем и вместо нее создаем новую частицу (фактически просто перезаписываем элемент)
                    p = particle();
        }


        // === Задание 4 ===
        if (!paused) {
            float A = 0.5, C = 0.1, D = 0.1;
            for (auto &p: particles) {  // для каждой вершины (обязательно по ссылке & итерируемся, чтобы p.<поле> = ... действительно меняло элемент вектора, а не меняло его копию)
                p.velocity.y += dt * A;  // наращиваем скорость по вертикали (идея в том, чтобы сделать пламя в последнем задании -> там частицы = кусочки пламени и они будут устремляться вверх все быстрее и быстрее)
                p.position += p.velocity * dt;  // наращиваем позицию (= интеграл от скорости)
                p.velocity *= exp(- C * dt);  // симулируем трение (об воздух)
                p.size *= exp(- D * dt);  // меняем размер (частицы огня уменьшаются и пропадают вверху)

                // === Задание 5 ===
                p.rotation += p.angular_velocity * dt;  // интегрируем угловую скокрость = получам приращение угла, его прибавили
            }
        }


        glBindBuffer(GL_ARRAY_BUFFER, vbo);
        glBufferData(GL_ARRAY_BUFFER, particles.size() * sizeof(particle), particles.data(), GL_STATIC_DRAW);

        glUseProgram(program);

        glUniformMatrix4fv(model_location, 1, GL_FALSE, reinterpret_cast<float *>(&model));
        glUniformMatrix4fv(view_location, 1, GL_FALSE, reinterpret_cast<float *>(&view));
        glUniformMatrix4fv(projection_location, 1, GL_FALSE, reinterpret_cast<float *>(&projection));
        glUniform3fv(camera_position_location, 1, reinterpret_cast<float *>(&camera_position));
        
        glBindVertexArray(vao);


        // === Задание 7 ===
        glEnable(GL_BLEND);  // Включаем аддитивный блендинг
        glBlendFunc(GL_SRC_ALPHA, GL_ONE);
        glDisable(GL_DEPTH_TEST);  // Отключаем тест глубины
        glBindTexture(GL_TEXTURE_2D, texture);  // Делаем текстуру текщу в 0-м (по умолчанию активный) texture unit, чтобы она попала в sampler2D

        // === Задание 8 ===
        glActiveTexture(GL_TEXTURE1);
        glBindTexture(GL_TEXTURE_1D, colorPaletteTexture);  // В 1-ом texture unit делаем текущей текстур с цветом


        glDrawArrays(GL_POINTS, 0, particles.size());

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
