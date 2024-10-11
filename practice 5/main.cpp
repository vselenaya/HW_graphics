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

uniform mat4 viewmodel;
uniform mat4 projection;

layout (location = 0) in vec3 in_position;
layout (location = 1) in vec3 in_normal;

out vec3 normal;

void main()
{
    gl_Position = projection * viewmodel * vec4(in_position, 1.0);
    normal = mat3(viewmodel) * in_normal;
}
)";

const char fragment_shader_source[] =
R"(#version 330 core

in vec3 normal;

layout (location = 0) out vec4 out_color;

void main()
{
    float lightness = 0.5 + 0.5 * dot(normalize(normal), normalize(vec3(1.0, 2.0, 3.0)));
    vec3 albedo = vec3(1.0);
    out_color = vec4(lightness * albedo, 1.0);
}
)";
*/



/*
// === Задание 2 ===
const char vertex_shader_source[] =
R"(#version 330 core

uniform mat4 viewmodel;
uniform mat4 projection;

layout (location = 0) in vec3 in_position;
layout (location = 1) in vec3 in_normal;
layout (location = 2) in vec2 in_texcoord;  // добавили переменную для аттрибута (индекс 2), соответствующий координатам в текстуре

out vec3 normal;
out vec2 texcoord;  // добавляем выходную переменную (которая передастся во фрагментный шейд)

void main()
{
    gl_Position = projection * viewmodel * vec4(in_position, 1.0);
    normal = mat3(viewmodel) * in_normal;
    texcoord = in_texcoord;  // просто передаём значение аттрибута (то есть координаты в текстуре) во фрагментный шейдер 
}
)";

const char fragment_shader_source[] =
R"(#version 330 core

in vec3 normal;
in vec2 texcoord;  // получаем координаты в текстуре из вершинного шейдера (они тоже (как и другие входные переменные) интерполируются для каждого пикселя (от которого и вызывается данный фрагментный шейдер) между вершинами )

layout (location = 0) out vec4 out_color;

void main()
{
    float lightness = 0.5 + 0.5 * dot(normalize(normal), normalize(vec3(1.0, 2.0, 3.0)));
    vec3 albedo = vec3(texcoord, 0.0);  // используем координаты в текстуре и как цвет
    out_color = vec4(lightness * albedo, 1.0);
}
)";
*/



/*
// === Задание 3 ===
const char vertex_shader_source[] =
R"(#version 330 core

uniform mat4 viewmodel;
uniform mat4 projection;

layout (location = 0) in vec3 in_position;
layout (location = 1) in vec3 in_normal;
layout (location = 2) in vec2 in_texcoord;  

out vec3 normal;
out vec2 texcoord; 

void main()
{
    gl_Position = projection * viewmodel * vec4(in_position, 1.0);
    normal = mat3(viewmodel) * in_normal;
    texcoord = in_texcoord;
}
)";

const char fragment_shader_source[] =
R"(#version 330 core

uniform sampler2D sampler;  // создаём uniform-переменную для семплера 2d-nекстуры

in vec3 normal;
in vec2 texcoord;  

layout (location = 0) out vec4 out_color;

void main()
{
    float lightness = 0.5 + 0.5 * dot(normalize(normal), normalize(vec3(1.0, 2.0, 3.0)));
    vec4 albedo = texture(sampler, texcoord);  // sampler вычитывает из текстуры цвет по нужным координатам - в формете r,g,b,a
    out_color = vec4(lightness * albedo.xyz, albedo.w);  // берём первые три координаты (через .xyz) (это цвет rgb), умножаем на какой-то ннеизвестный пока параетр; в конце добавляем параметр прозрачности - четвёртая координата (через .w)
}
)";
*/



// === Задание 6 ===
const char vertex_shader_source[] =
R"(#version 330 core

uniform mat4 viewmodel;
uniform mat4 projection;

layout (location = 0) in vec3 in_position;
layout (location = 1) in vec3 in_normal;
layout (location = 2) in vec2 in_texcoord;  

out vec3 normal;
out vec2 texcoord; 

void main()
{
    gl_Position = projection * viewmodel * vec4(in_position, 1.0);
    normal = mat3(viewmodel) * in_normal;
    texcoord = in_texcoord;
}
)";

const char fragment_shader_source[] =
R"(#version 330 core

uniform sampler2D sampler; 
uniform float time;  // добавляем переменную времени извне

in vec3 normal;
in vec2 texcoord;  

layout (location = 0) out vec4 out_color;

void main()
{
    float lightness = 0.5 + 0.5 * dot(normalize(normal), normalize(vec3(1.0, 2.0, 3.0)));
    vec4 albedo = texture(sampler, texcoord + vec2(time, sin(time)));  // добавляем к координатам зависящее от времени что-то  
                                                                       // заметим, что при таком добавлении координаты вылезут за пределы текстуры, но у нас по умолчанию включен wrapping mode = GL_REPEAT, который просто циклично дублирует текстуру -> ничего не ломается
    out_color = vec4(lightness * albedo.xyz, albedo.w);  
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

    GLuint viewmodel_location = glGetUniformLocation(program, "viewmodel");
    GLuint projection_location = glGetUniformLocation(program, "projection");

    std::string project_root = PROJECT_ROOT;
    std::string cow_texture_path = project_root + "/cow.png";
    obj_data cow = parse_obj(project_root + "/cow.obj");




    // =========  МОЙ КОД ================


    // === Задание 1 === (полностью как в предыдущей практике, но bunny на cow меняем)
    GLuint vbo, vao, ebo;

    glGenBuffers(1, &vbo);
    glBindBuffer(GL_ARRAY_BUFFER, vbo);
    glBufferData(GL_ARRAY_BUFFER, cow.vertices.size() * sizeof(obj_data::vertex), cow.vertices.data(), GL_STATIC_DRAW);

    glGenVertexArrays(1, &vao);
    glBindVertexArray(vao);
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, sizeof(obj_data::vertex), (void*)(0));
    glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, sizeof(obj_data::vertex), (void*)(sizeof(obj_data::vertex::position))); 
    glEnableVertexAttribArray(0); 
    glEnableVertexAttribArray(1);

    glGenBuffers(1, &ebo); 
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, ebo); 
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, cow.indices.size() * sizeof(uint32_t), cow.indices.data(), GL_STATIC_DRAW);



    // === Задание 2 ===
    glVertexAttribPointer(2, 2, GL_FLOAT, GL_FALSE, sizeof(obj_data::vertex), (void*)(sizeof(obj_data::vertex::position) + sizeof(obj_data::vertex::normal))); 
    glEnableVertexAttribArray(2);   // настроили аттрибут для координат текстуры (in_texcoord в вершинном шейд)



    // === Задание 3 ===
    GLuint texture;
    glGenTextures(1, &texture);  // создали объект текстуры
    glActiveTexture(GL_TEXTURE0 + 0);  // делаем текущим texture unite с номером 0 (можно и не делать, так как по умолчанию 0-ой и так включен... но на всякий случай)
    glBindTexture(GL_TEXTURE_2D, texture);  // в рамках этого texture unite делаем текущей нашу текстуру, причём приписываем её target = GL_TEXTURE_2D
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);  // её (текущую текстуру) настраиваем для случаев, когда она будет рисовать меньше и больше, чем есть на самом деле
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    
    int size = 512;  // задём размер (будем делать текстуру размера 512 на 512 пикселей - вот если на экране её будут, например, отрисовывать меньшее число пикселей, тогда как раз настройка GL_TEXTURE_MIN_FILTER отработает)
    std::vector<std::uint32_t> pixels(size * size);  // создали вектор для цветов каждого из 512 * 512 пикселей текстуры (каждый цвет - это четвёрка R(значение красного канала),G(зелёного канала),B(снего),A(значение прозрачности), причём каждое из этих значений занимает 1 байт = 8бит -> цвет каждого пискеля = 4 * 8 = 32 бит -> тип данных для цвета uint32)
    // ВАЖНО!!! в OpenGL всегда подразумевается little-endian формат - то есть цвет RGBA должен записываться от младшего байта (отвечающего за R) к старшему, 
    // то есть в C++ запись 32-битного числа для RGBA будет выглядеть как: 0xAABBGGRR (где AA(старший байт),BB,GG,RR(младший байт) - это байты (в виде двузначного 16тиричного числа - как раз со значениями от 0 до 255) для alpha-канала и цветов), то есть, например, 0xFF0000FF - это непрозрачный красный; сами байты всегда записываются от старшего к младшему - например, A0 - это 160, FF=255, 08 = 8.
    for (size_t i = 0; i < size; i ++) {
        for (size_t j = 0; j < size; j ++) {  // рассматриваем (i,j)-ый пиксель текстуры (ему отвечает элемент pixels с индексом i*size+j)
        if ((i+j) % 2 == 0)
            pixels[i * size + j] = 0xFF000000u;  // в шахматном порядке раскрашваем пиксель в чёрный (значения в каждом канале R,G,B равно 0, только значение alpha-кнала прозрачности = 255 (=макс значение) (непрозрачный))
        else
            pixels[i * size + j] = 0xFFFFFFFFu;  // и белый (все каналы равны 255)
        }
    }
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, size, size, 0, GL_RGBA, GL_UNSIGNED_BYTE, pixels.data());  // загружаем данные (цвета пикселей) для нашей текстуры (той, которая текущая с target=GL_TEXTURE_2D); 
    GLuint sampler_location = glGetUniformLocation(program, "sampler");  // получаем указатель на переменную sampler в шейдере



    // === Задание 4 === 
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST_MIPMAP_NEAREST);  // меняем фильтр - в этом случае будут использоваться mipmap-уровни (а значит они обязательно должны быть)
    glGenerateMipmap(GL_TEXTURE_2D);  // вызываем команду, чтобы для текущей текстуры (с таргетом GL_TEXTURE_2d) автоматически сгенерировались все mipmap-уровни (номера которых от 1 до log_2(size), а размеры от size/2 до 1)
    std::vector<std::uint32_t> mono_red(size/2 * size/2, 0xFF0000FFu);  // создаём вектор из полностью красных цветов (каждый цвет имеет макс значение по R-канлу (и по A-кналу, чтобы непрозрачный) и 0 у остальных) -> этим вектором перезапишем цвета 1-го mipmap-уровня нашей текстуры (размер этого уровня ровно в 2 раза меньше по каждой оси -> в вектор делаем size/2)
    std::vector<std::uint32_t> mono_green(size/4 * size/4, 0xFF00FF00u);  // аналогично полностью зелёный цвет для 2-го уровня (там размер /4)
    std::vector<std::uint32_t> mono_blue(size/8 * size/8, 0xFFFF0000u);  // полностью синий для 3-его уровня (размер /8)
    glTexImage2D(GL_TEXTURE_2D, 1, GL_RGBA8, size/2, size/2, 0, GL_RGBA, GL_UNSIGNED_BYTE, mono_red.data());
    glTexImage2D(GL_TEXTURE_2D, 2, GL_RGBA8, size/4, size/4, 0, GL_RGBA, GL_UNSIGNED_BYTE, mono_green.data());  // подменяем цвет каждого из 1,2,3 mipmap уровней
    glTexImage2D(GL_TEXTURE_2D, 3, GL_RGBA8, size/8, size/8, 0, GL_RGBA, GL_UNSIGNED_BYTE, mono_blue.data());   // (важно было сгенерировать уровни glGenerateMipmap, так как помимо этих трёх есть ещё логарифм более мельких уровней, которые обязательно должны быть, иначе картинка будет чёрной считаться, как на лекци говорили)
    // вы вывод: при удалении камеры от объекта (когда он становится всё мельче), текстура занимет всё меньше пикселей - используется более низкий mimap уровень -> больше объекта становится синей



    // === Задание 5 ===
    GLuint new_texture;
    glGenTextures(1, &new_texture);  // создаёем новую текстуру
    glActiveTexture(GL_TEXTURE0 + 1); 
    glBindTexture(GL_TEXTURE_2D, new_texture);  // её делаем текущей уже в texture unit номером 1 (для этого переключаемся на него); снова такргет такой же как раньше
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_LINEAR); 
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);

    int x, y, channels;  // создали переменные, которые заполянтся при считывании изображения
    stbi_uc *img = stbi_load(cow_texture_path.data(), &x, &y, &channels, 4);  // считываем изображение (текстуру) из файла билиотекой stb_image (весь код которой удобно доступен из header .h) -> нам возвращается stbi_uc* указатель на выделенную память под считанное изображение, а также заполнились переменные x и y размера изображения
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, x, y, 0, GL_RGBA, GL_UNSIGNED_BYTE, img);  // загружаем в текущую (в текущую в рамках texture unit номер 1, который сейчас сделан активным) текстуру (с таргет = GL_TEXTURE_2D - то есть в текстуру new_texture, которая сделана текущй с таким таргетом и в texture unit 1) внутрь GPU данные изображения 
    stbi_image_free(img);  // очищаем память из-под изображения
    glGenerateMipmap(GL_TEXTURE_2D);  // у этой же текстуры автоматически генерируем mimpa-уровни



    // === Задание 6 ===
    GLuint time_location = glGetUniformLocation(program, "time");



    auto last_frame_start = std::chrono::high_resolution_clock::now();
    float time = 0.f;
    float angle_y = M_PI;
    float offset_z = -2.f;

    std::map<SDL_Keycode, bool> button_down;

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

        if (button_down[SDLK_UP]) offset_z -= 4.f * dt;
        if (button_down[SDLK_DOWN]) offset_z += 4.f * dt;
        if (button_down[SDLK_LEFT]) angle_y += 4.f * dt;
        if (button_down[SDLK_RIGHT]) angle_y -= 4.f * dt;

        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
        glEnable(GL_DEPTH_TEST);

        float near = 0.1f;
        float far = 100.f;
        float top = near;
        float right = (top * width) / height;

        float viewmodel[16] =
        {
            std::cos(angle_y), 0.f, -std::sin(angle_y), 0.f,
            0.f, 1.f, 0.f, 0.f,
            std::sin(angle_y), 0.f, std::cos(angle_y), offset_z,
            0.f, 0.f, 0.f, 1.f,
        };

        float projection[16] =
        {
            near / right, 0.f, 0.f, 0.f,
            0.f, near / top, 0.f, 0.f,
            0.f, 0.f, - (far + near) / (far - near), - 2.f * far * near / (far - near),
            0.f, 0.f, -1.f, 0.f,
        };

        glUseProgram(program);
        glUniformMatrix4fv(viewmodel_location, 1, GL_TRUE, viewmodel);
        glUniformMatrix4fv(projection_location, 1, GL_TRUE, projection);



        // === Задание 6 ===
        glUniform1f(time_location, time);  // передаём в шейдер


        // === Задание 3 ===
        glUniform1i(sampler_location, 0);  // передаём в переменную 0 (номер texture unit, из котрой нужно взять текущую текстуру - в данном случае sampler имеет тип 2d, поэтому он возьмёт текущую текстуру с таргетом = GL_TEXTURE_2D)
        glActiveTexture(GL_TEXTURE0 + 0);
        glBindTexture(GL_TEXTURE_2D, texture);  // на всякий случай ещё раз делаем нашу текстуру (у которой таргет = GL_TEXTURE_2D!!!! - помним, что после того как один раз сделали bind  текстуры с каким-то таргетом, его нельяз потом менять) текущей для texture unit номер 0


        // === Задание 5 === - тут мы перезатираем данные из предыдущих 3ёх строк задания 3
        glUniform1i(sampler_location, 1);  // передаем в шейдер номер texture unit - в нашем случа 1 и больше ничего не нужно делать (так как в данном texture unit в качестве текущей текстуры с таргет = GL_TEXTURE_2D уже установлена нужная нам new_texture - и не важно, 
                                           // что в предыдущих строчках мы делаем активной 0ой texture unit..., главное что ранее new_texture была сделана текущей в texture unit 1 и никакую другую текстуру с таргетом  = GL_TEXTURE_2D мы не делали текущей внутри именно в 1ой texture unit; аналогично и предыдущие строки "glActiveTexture(GL_TEXTURE0 + 0); и glBindTexture(GL_TEXTURE_2D, texture);" для задание 3 не нужны, так как в 0ом texture unit уже сделана текущей texture и других после этого текущими не делали...)
                                           // итог: внутри каждого texture unit запомнена для каждого target последняя сделанная текущей текстура -> когда внутрь шейдера передаётся номер texture unit, шейдер работает с текущей текстурой (для target, который требуется конкретному sampler) -> тут можно передавать 0 или 1 и в зависимости от этого будет использвана или текстура texture из задания 4, или текстура new_texture из задания 5


        // === Задание 1 ===
        glBindVertexArray(vao);
        glDrawElements(GL_TRIANGLES, cow.indices.size(), GL_UNSIGNED_INT, (void*)0);  // просто отрисовываем треуголники коровы cow

        
        SDL_GL_SwapWindow(window);  // это команда показывает, что настройка и отправка данных окончена, можно рисовать
    }

    SDL_GL_DeleteContext(gl_context);
    SDL_DestroyWindow(window);
}
catch (std::exception const & e)
{
    std::cerr << e.what() << std::endl;
    return EXIT_FAILURE;
}
