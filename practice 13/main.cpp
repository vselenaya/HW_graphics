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

#define GLM_FORCE_SWIZZLE
#define GLM_ENABLE_EXPERIMENTAL
#include <glm/vec3.hpp>
#include <glm/mat4x4.hpp>
#include <glm/ext/matrix_transform.hpp>
#include <glm/ext/matrix_clip_space.hpp>
#include <glm/ext/scalar_constants.hpp>
#include <glm/gtx/quaternion.hpp>
#include <glm/gtx/string_cast.hpp>

#include "gltf_loader.hpp"
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

uniform mat4 model;
uniform mat4 view;
uniform mat4 projection;

layout (location = 0) in vec3 in_position;
layout (location = 1) in vec3 in_normal;
layout (location = 2) in vec2 in_texcoord;

out vec3 normal;
out vec2 texcoord;

void main()
{

    gl_Position = projection * view * model * vec4(in_position, 1.0);
    normal = mat3(model) * in_normal;
    texcoord = in_texcoord;
}
)";
*/

/*
// === Задание 1 ===
const char vertex_shader_source[] =
R"(#version 330 core

uniform mat4 model;
uniform mat4 view;
uniform mat4 projection;

layout (location = 0) in vec3 in_position;
layout (location = 1) in vec3 in_normal;
layout (location = 2) in vec2 in_texcoord;
layout (location = 3) in ivec4 in_joints; // Номера костей
layout (location = 4) in vec4 in_weights; // Веса костей

out vec3 normal;
out vec2 texcoord;
out vec4 weights; // Передаем веса во фрагментный шейдер

void main()
{

    gl_Position = projection * view * model * vec4(in_position, 1.0);
    normal = mat3(model) * in_normal;
    texcoord = in_texcoord;
    weights = in_weights; // Передаем веса
}
)";
*/


// === Задание 2 ===
const char vertex_shader_source[] =
R"(#version 330 core

uniform mat4 model;
uniform mat4 view;
uniform mat4 projection;
uniform mat4x3 bones[100]; // Массив матриц костей

layout (location = 0) in vec3 in_position;
layout (location = 1) in vec3 in_normal;
layout (location = 2) in vec2 in_texcoord;
layout (location = 3) in ivec4 in_joints;
layout (location = 4) in vec4 in_weights;

out vec3 normal;
out vec2 texcoord;
out vec4 weights;

void main()
{
    mat4x3 average = mat4x3(0.0);  // Инициализация матрицы

    // Вычисляем взвешенное среднее матриц
    for (int i = 0; i < 4; ++i) {
        average += bones[in_joints[i]] * in_weights[i];       
    }


    gl_Position = projection * view * model * mat4(average) * vec4(in_position, 1.0);  // применяем average перед другими...
    normal = mat3(model) * mat3(average) * in_normal;
    texcoord = in_texcoord;
    weights = in_weights; // Передаем веса
}
)";


/* было изначально
const char fragment_shader_source[] =
R"(#version 330 core

uniform sampler2D albedo;
uniform vec4 color;
uniform int use_texture;

uniform vec3 light_direction;

layout (location = 0) out vec4 out_color;

in vec3 normal;
in vec2 texcoord;

void main()
{
    vec4 albedo_color;

    if (use_texture == 1)
        albedo_color = texture(albedo, texcoord);
    else
        albedo_color = color;

    float ambient = 0.4;
    float diffuse = max(0.0, dot(normalize(normal), light_direction));

    out_color = vec4(albedo_color.rgb * (ambient + diffuse), albedo_color.a);
}
)";
*/

/*
// === Задание 1 ===
const char fragment_shader_source[] =
R"(#version 330 core

uniform sampler2D albedo;
uniform vec4 color;
uniform int use_texture;

uniform vec3 light_direction;

layout (location = 0) out vec4 out_color;

in vec3 normal;
in vec2 texcoord;
in vec4 weights; // Получаем веса из вершинного шейдера

void main()
{
    vec4 albedo_color;

    if (use_texture == 1)
        albedo_color = texture(albedo, texcoord);
    else
        albedo_color = color;

    float ambient = 0.4;
    float diffuse = max(0.0, dot(normalize(normal), light_direction));

    out_color = weights;  // Используем веса в качестве цвета
}
)";
*/


// === Задание 1 ===
const char fragment_shader_source[] =
R"(#version 330 core

uniform sampler2D albedo;
uniform vec4 color;
uniform int use_texture;

uniform vec3 light_direction;

layout (location = 0) out vec4 out_color;

in vec3 normal;
in vec2 texcoord;
in vec4 weights;

void main()
{
    vec4 albedo_color;

    if (use_texture == 1)
        albedo_color = texture(albedo, texcoord);
    else
        albedo_color = color;

    float ambient = 0.4;
    float diffuse = max(0.0, dot(normalize(normal), light_direction));

    out_color = vec4(albedo_color.rgb * (ambient + diffuse), albedo_color.a);  // возвращаем цвет
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

int main() try
{
    if (SDL_Init(SDL_INIT_VIDEO) != 0)
        sdl2_fail("SDL_Init: ");

    SDL_GL_SetAttribute(SDL_GL_CONTEXT_MAJOR_VERSION, 3);
    SDL_GL_SetAttribute(SDL_GL_CONTEXT_MINOR_VERSION, 3);
    SDL_GL_SetAttribute(SDL_GL_CONTEXT_PROFILE_MASK, SDL_GL_CONTEXT_PROFILE_CORE);
    SDL_GL_SetAttribute(SDL_GL_DOUBLEBUFFER, 1);
    SDL_GL_SetAttribute(SDL_GL_MULTISAMPLEBUFFERS, 1);
    SDL_GL_SetAttribute(SDL_GL_MULTISAMPLESAMPLES, 16);
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

    auto vertex_shader = create_shader(GL_VERTEX_SHADER, vertex_shader_source);
    auto fragment_shader = create_shader(GL_FRAGMENT_SHADER, fragment_shader_source);
    auto program = create_program(vertex_shader, fragment_shader);

    GLuint model_location = glGetUniformLocation(program, "model");
    GLuint view_location = glGetUniformLocation(program, "view");
    GLuint projection_location = glGetUniformLocation(program, "projection");
    GLuint albedo_location = glGetUniformLocation(program, "albedo");
    GLuint color_location = glGetUniformLocation(program, "color");
    GLuint use_texture_location = glGetUniformLocation(program, "use_texture");
    GLuint light_direction_location = glGetUniformLocation(program, "light_direction");

    const std::string project_root = PROJECT_ROOT;
    const std::string model_path = project_root + "/dancing/dancing.gltf";

    auto const input_model = load_gltf(model_path);
    GLuint vbo;
    glGenBuffers(1, &vbo);
    glBindBuffer(GL_ARRAY_BUFFER, vbo);
    glBufferData(GL_ARRAY_BUFFER, input_model.buffer.size(), input_model.buffer.data(), GL_STATIC_DRAW);

    struct mesh
    {
        GLuint vao;
        gltf_model::accessor indices;
        gltf_model::material material;
    };

    auto setup_attribute = [](int index, gltf_model::accessor const & accessor, bool integer = false)
    {
        glEnableVertexAttribArray(index);
        if (integer)
            glVertexAttribIPointer(index, accessor.size, accessor.type, 0, reinterpret_cast<void *>(accessor.view.offset));
        else
            glVertexAttribPointer(index, accessor.size, accessor.type, GL_FALSE, 0, reinterpret_cast<void *>(accessor.view.offset));
    };

    std::vector<mesh> meshes;
    for (auto const & mesh : input_model.meshes)
    {
        for (auto const & primitive : mesh.primitives)
        {
            auto & result = meshes.emplace_back();
            glGenVertexArrays(1, &result.vao);
            glBindVertexArray(result.vao);

            glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, vbo);
            result.indices = primitive.indices;

            setup_attribute(0, primitive.position);
            setup_attribute(1, primitive.normal);
            setup_attribute(2, primitive.texcoord);
            setup_attribute(3, primitive.joints, true);
            setup_attribute(4, primitive.weights);

            result.material = primitive.material;
        }
    }

    std::map<std::string, GLuint> textures;
    for (auto const & mesh : meshes)
    {
        if (!mesh.material.texture_path) continue;
        if (textures.contains(*mesh.material.texture_path)) continue;

        auto path = std::filesystem::path(model_path).parent_path() / *mesh.material.texture_path;

        int width, height, channels;
        auto data = stbi_load(path.c_str(), &width, &height, &channels, 4);
        assert(data);

        GLuint texture;
        glGenTextures(1, &texture);
        glBindTexture(GL_TEXTURE_2D, texture);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_LINEAR);
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, width, height, 0, GL_RGBA, GL_UNSIGNED_BYTE, data);
        glGenerateMipmap(GL_TEXTURE_2D);

        stbi_image_free(data);

        textures[*mesh.material.texture_path] = texture;
    }

    auto last_frame_start = std::chrono::high_resolution_clock::now();

    float time = 0.f;

    std::map<SDL_Keycode, bool> button_down;

    float view_angle = 0.f;
    float camera_distance = 1.5f;

    float camera_rotation = 0.f;
    float camera_height = 1.f;

    bool paused = false;


    // === Задание 2 ===
    GLuint bones_location = glGetUniformLocation(program, "bones");

    // === Задание 5 ===
    std::string curr_anim = "hip-hop";
    std::string prev_anim = "hip-hop";
    float from_prev_time = 10;

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

        if (!paused)
            time += dt;

        if (button_down[SDLK_UP])
            camera_distance -= 3.f * dt;
        if (button_down[SDLK_DOWN])
            camera_distance += 3.f * dt;

        if (button_down[SDLK_a])
            camera_rotation -= 2.f * dt;
        if (button_down[SDLK_d])
            camera_rotation += 2.f * dt;

        if (button_down[SDLK_w])
            view_angle -= 2.f * dt;
        if (button_down[SDLK_s])
            view_angle += 2.f * dt;

        glClearColor(0.8f, 0.8f, 1.f, 0.f);
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

        glEnable(GL_DEPTH_TEST);

        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

        float near = 0.1f;
        float far = 100.f;


        /* было изначально
        glm::mat4 model = glm::scale(glm::mat4(1.f), glm::vec3(1.f));
        */

        // === Задание 3 ===
        glm::mat4 model = glm::scale(glm::mat4(1.f), glm::vec3(0.01f));


        glm::mat4 view(1.f);
        view = glm::translate(view, {0.f, 0.f, -camera_distance});
        view = glm::rotate(view, view_angle, {1.f, 0.f, 0.f});
        view = glm::rotate(view, camera_rotation, {0.f, 1.f, 0.f});
        view = glm::translate(view, {0.f, -camera_height, 0.f});

        glm::mat4 projection = glm::perspective(glm::pi<float>() / 2.f, (1.f * width) / height, near, far);

        glm::vec3 camera_position = (glm::inverse(view) * glm::vec4(0.f, 0.f, 0.f, 1.f)).xyz();

        glm::vec3 light_direction = glm::normalize(glm::vec3(1.f, 2.f, 3.f));

        glUseProgram(program);
        glUniformMatrix4fv(model_location, 1, GL_FALSE, reinterpret_cast<float *>(&model));
        glUniformMatrix4fv(view_location, 1, GL_FALSE, reinterpret_cast<float *>(&view));
        glUniformMatrix4fv(projection_location, 1, GL_FALSE, reinterpret_cast<float *>(&projection));
        glUniform3fv(light_direction_location, 1, reinterpret_cast<float *>(&light_direction));

        
        /*
        // === Задание 2 ===
        float scale = 0.75f + cos(time) * 0.25f;  // Изменяющаяся переменная scale
        std::vector<glm::mat4x3> bone_matrices(input_model.bones.size(), glm::mat4x3(scale)); // Инициализация массива матриц
        glUniformMatrix4x3fv(bones_location, bone_matrices.size(), GL_FALSE, reinterpret_cast<float*>(bone_matrices.data()));  // Передача матриц в uniform-массив
        */


        /*
        // === Задание 3 ===
        std::string dance_name = "hip-hop";
        const auto& animation = input_model.animations.at(dance_name);  // Получаем анимацию "hip-hop" !!! важно at, а не [], так как у const нельзя []
        std::vector<glm::mat4x3> bone_matrices(input_model.bones.size()); // Инициализация массива матриц
        
        // Вычисляем преобразования для каждой кости
        for (size_t i = 0; i < input_model.bones.size(); i ++) {

            // Получаем трансляцию, вращение и масштаб
            
            //glm::vec3 translation = animation.bones[i].translation(0.f);
            //glm::quat rotation = animation.bones[i].rotation(0.f);
            //glm::vec3 scale = animation.bones[i].scale(0.f);
            
            // === Задание 4 ===
            glm::vec3 translation = animation.bones[i].translation(std::fmod(time, animation.max_time));
            glm::quat rotation = animation.bones[i].rotation(std::fmod(time, animation.max_time));
            glm::vec3 scale = animation.bones[i].scale(std::fmod(time, animation.max_time));

            // Вычисляем матрицы
            glm::mat4 translation_matrix = glm::translate(glm::mat4(1.f), translation);
            glm::mat4 rotation_matrix = glm::toMat4(rotation);
            glm::mat4 scale_matrix = glm::scale(glm::mat4(1.f), scale);

            // Вычисляем общее преобразование
            glm::mat4 transform = translation_matrix * rotation_matrix * scale_matrix;

            // Если у кости есть родитель, домножаем на матрицу родителя
            if (input_model.bones[i].parent != -1) {
                auto parent_transform = bone_matrices[input_model.bones[i].parent];  // для родителя уже должно быть все посчитано (так как идёт слева направо по возрастанию i, а данные устроены так, что кость-родитель всегда имеет номер меньше самой кости)
                transform = parent_transform * transform;  // Умножаем на матрицу родителя
            }

            bone_matrices[i] = glm::mat4x3(transform);  // Конвертируем в mat4x3
        }

        // После этого домножаем каждую кость на её inverse-bind матрицу
        for (size_t i = 0; i < input_model.bones.size(); i ++) {
            bone_matrices[i] = bone_matrices[i] * input_model.bones[i].inverse_bind_matrix;
        }

        glUniformMatrix4x3fv(bones_location, bone_matrices.size(), GL_FALSE, reinterpret_cast<float*>(bone_matrices.data()));
        */



        // === Задание 5 ===
        from_prev_time += dt;

        if (button_down[SDLK_1] && from_prev_time > 1.f) {
            prev_anim = curr_anim;
            curr_anim = "hip-hop";
            from_prev_time = 0.f;
        } else if (button_down[SDLK_2] && from_prev_time > 1.f) {
            prev_anim = curr_anim;
            curr_anim = "rumba";
            from_prev_time = 0.f;
        } else if (button_down[SDLK_3] && from_prev_time > 1.f) {
            prev_anim = curr_anim;
            curr_anim = "flair";
            from_prev_time = 0.f;
        }

        const auto& animation_curr = input_model.animations.at(curr_anim);
        const auto& animation_prev = input_model.animations.at(prev_anim);
        std::vector<glm::mat4x3> bone_matrices(input_model.bones.size());
        
        for (size_t i = 0; i < input_model.bones.size(); i ++) {
            // Получаем для текущего и предыдущего движения:
            glm::vec3 translation_curr = animation_curr.bones[i].translation(std::fmod(time, animation_curr.max_time));
            glm::quat rotation_curr = animation_curr.bones[i].rotation(std::fmod(time, animation_curr.max_time));
            glm::vec3 scale_curr = animation_curr.bones[i].scale(std::fmod(time, animation_curr.max_time));
            glm::vec3 translation_prev = animation_prev.bones[i].translation(std::fmod(time, animation_prev.max_time));
            glm::quat rotation_prev = animation_prev.bones[i].rotation(std::fmod(time, animation_prev.max_time));
            glm::vec3 scale_prev = animation_prev.bones[i].scale(std::fmod(time, animation_prev.max_time));

            // В течении 1 секунды после смены, интерполируем (= коэффициент = время, прошедшее со смены)
            glm::vec3 translation = glm::lerp(translation_prev, translation_curr, std::min(1.f, from_prev_time));
            glm::quat rotation = glm::slerp(rotation_prev, rotation_curr, std::min(1.f, from_prev_time));
            glm::vec3 scale = glm::lerp(scale_prev, scale_curr, std::min(1.f, from_prev_time));

            // Вычисляем матрицы
            glm::mat4 translation_matrix = glm::translate(glm::mat4(1.f), translation);
            glm::mat4 rotation_matrix = glm::toMat4(rotation);
            glm::mat4 scale_matrix = glm::scale(glm::mat4(1.f), scale);

            // Вычисляем общее преобразование
            glm::mat4 transform = translation_matrix * rotation_matrix * scale_matrix;

            // Если у кости есть родитель, домножаем на матрицу родителя
            if (input_model.bones[i].parent != -1) {
                auto parent_transform = bone_matrices[input_model.bones[i].parent];  // для родителя уже должно быть все посчитано (так как идёт слева направо по возрастанию i, а данные устроены так, что кость-родитель всегда имеет номер меньше самой кости)
                transform = parent_transform * transform;  // Умножаем на матрицу родителя
            }

            bone_matrices[i] = glm::mat4x3(transform);  // Конвертируем в mat4x3
        }

        // После этого домножаем каждую кость на её inverse-bind матрицу
        for (size_t i = 0; i < input_model.bones.size(); i ++) {
            bone_matrices[i] = bone_matrices[i] * input_model.bones[i].inverse_bind_matrix;
        }

        glUniformMatrix4x3fv(bones_location, bone_matrices.size(), GL_FALSE, reinterpret_cast<float*>(bone_matrices.data()));



        auto draw_meshes = [&](bool transparent)
        {
            for (auto const & mesh : meshes)
            {
                if (mesh.material.transparent != transparent)
                    continue;

                if (mesh.material.two_sided)
                    glDisable(GL_CULL_FACE);
                else
                    glEnable(GL_CULL_FACE);

                if (transparent)
                    glEnable(GL_BLEND);
                else
                    glDisable(GL_BLEND);

                if (mesh.material.texture_path)
                {
                    glBindTexture(GL_TEXTURE_2D, textures[*mesh.material.texture_path]);
                    glUniform1i(use_texture_location, 1);
                }
                else if (mesh.material.color)
                {
                    glUniform1i(use_texture_location, 0);
                    glUniform4fv(color_location, 1, reinterpret_cast<const float *>(&(*mesh.material.color)));
                }
                else
                    continue;

                glBindVertexArray(mesh.vao);
                glDrawElements(GL_TRIANGLES, mesh.indices.count, mesh.indices.type, reinterpret_cast<void *>(mesh.indices.view.offset));
            }
        };

        draw_meshes(false);
        glDepthMask(GL_FALSE);
        draw_meshes(true);
        glDepthMask(GL_TRUE);

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
