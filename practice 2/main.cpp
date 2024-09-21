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
#include <unordered_map>


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


/* === было исходно ===
const char vertex_shader_source[] =
R"(#version 330 core

const vec2 VERTICES[3] = vec2[3](
    vec2(0.0, 1.0),
    vec2(-sqrt(0.75), -0.5),
    vec2( sqrt(0.75), -0.5)
);

const vec3 COLORS[3] = vec3[3](
    vec3(1.0, 0.0, 0.0),
    vec3(0.0, 1.0, 0.0),
    vec3(0.0, 0.0, 1.0)
);

out vec3 color;

void main()
{
    vec2 position = VERTICES[gl_VertexID];
    gl_Position = vec4(position, 0.0, 1.0);
    color = COLORS[gl_VertexID];
}
)";
*/


const char fragment_shader_source[] =
R"(#version 330 core

in vec3 color;

layout (location = 0) out vec4 out_color;

void main()
{
    out_color = vec4(color, 1.0);
}
)";


/*
// === Задание 1 и 2: ===
const char vertex_shader_source[] =
R"(#version 330 core

uniform float scale;  // извне приходит масштаб (во сколько раз нужно увеличить изображение треугольника)
uniform float angle;  // извне приходит угол, на который нужно повернуть треугольник

const vec2 VERTICES[3] = vec2[3](
    vec2(0.0, 1.0),
    vec2(-sqrt(0.75), -0.5),
    vec2( sqrt(0.75), -0.5)
);  // координаты вершин треугольника = координаты векторов, выходящих из начала координат в вершины треугольника

vec2 center = 1/3 * VERTICES[0] + 1/3 * VERTICES[1] + 1/3 * VERTICES[2];  // считаем координаты центра треугольника (= вектор в него)
mat2 rotating_matrix = mat2(cos(angle), -sin(angle),
                            sin(angle), cos(angle));  // формируем матрицу поворота из данного угла angle
vec2 rotated_scaled_vertices[3] = vec2[3](
    scale * rotating_matrix * (VERTICES[0] - center) + center,
    scale * rotating_matrix * (VERTICES[1] - center) + center,
    scale * rotating_matrix * (VERTICES[2] - center) + center
);  // доворачиваем и меняем масштаб (делая VERTICES[i]-center, мы переходим в систему отсчёта = центр треугольника -> логично, ведь хотим поворачивать относительно него)

const vec3 COLORS[3] = vec3[3](
    vec3(1.0, 0.0, 0.0),
    vec3(0.0, 1.0, 0.0),
    vec3(0.0, 0.0, 1.0)
);

out vec3 color;  // выходное значение, которое попадёт в пиксельный шейдер (для каждого пикселя будет выдано интерполированное значение)

void main()
{   
    vec2 position = rotated_scaled_vertices[gl_VertexID];
    gl_Position = vec4(position, 0.0, 1.0);
    color = COLORS[gl_VertexID];
}
)";
*/


/*
// === Задание 3, 4, 5, 6: ===
const char vertex_shader_source[] =
R"(#version 330 core

uniform mat4 transform;  // извне теперь сразу вся матрица преобразований
uniform mat4 view;  // задание 5

const vec2 VERTICES[3] = vec2[3](
    vec2(0.0, 1.0),
    vec2(-sqrt(0.75), -0.5),
    vec2( sqrt(0.75), -0.5)
); 

const vec3 COLORS[3] = vec3[3](
    vec3(1.0, 0.0, 0.0),
    vec3(0.0, 1.0, 0.0),
    vec3(0.0, 0.0, 1.0)
);

out vec3 color;

void main()
{   
    vec2 position = VERTICES[gl_VertexID];
    gl_Position = transform * vec4(position, 0.0, 1.0);  // умножаем матрицу на каждый вектор (= вершину VERTICES[i], которую дополнили число 0.0 (видимо глубина, для 3d...) и числом 1 (показывает, что это не вектор, а точка - проективные коордианты...))    color = COLORS[gl_VertexID];
    gl_Position = view * gl_Position;  // для задания 5 ещё умножаем на view, выравнивая соотношение координат
    color = COLORS[gl_VertexID];
}
)";
*/


// === Задание 7: ===
const char vertex_shader_source[] =
R"(#version 330 core

uniform mat4 transform; 
uniform mat4 view; 

const vec2 VERTICES[8] = vec2[8](
    vec2(0.0, 0.0),  // эта точка - центр шестиугольника
    vec2(1.0, 0.0),  // следующие 6 точек - его вершины (против часовой стрелки)
    vec2(cos(60 * 3.14/180), cos(30 * 3.14/180)),
    vec2(-cos(60 * 3.14/180), cos(30 * 3.14/180)),
    vec2(-1.0, 0.0),
    vec2(-cos(60 * 3.14/180), -cos(30 * 3.14/180)),
    vec2(cos(60 * 3.14/180), -cos(30 * 3.14/180)),
    vec2(1.0, 0.0)  // далее ещё раз пишем первую вершину шестиугольника
);  // в таком случае, если использовать ключ GL_TRIANGLE_FAN будет нарисован весь шестиугольник как 6 треугольников, выходящих из центра - см лекции

const vec3 COLORS[8] = vec3[8](
    vec3(1.0, 0.0, 0.0),
    vec3(0.0, 1.0, 0.0),
    vec3(0.0, 0.0, 1.0),
    vec3(0.0, 1.0, 0.0),
    vec3(0.0, 0.0, 1.0),
    vec3(0.0, 1.0, 0.0),
    vec3(0.0, 1.0, 0.0),
    vec3(1.0, 0.0, 0.0)
);  // какие-то цвета выставляем...

out vec3 color;

void main()
{   
    vec2 position = VERTICES[gl_VertexID];
    gl_Position = transform * vec4(position, 0.0, 1.0); 
    gl_Position = view * gl_Position; 
    color = COLORS[gl_VertexID];
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
    // === базовый код ===
    if (SDL_Init(SDL_INIT_VIDEO) != 0)
        sdl2_fail("SDL_Init: ");

    SDL_Window * window = SDL_CreateWindow("Graphics course practice 2",
        SDL_WINDOWPOS_CENTERED,
        SDL_WINDOWPOS_CENTERED,
        800, 600,
        SDL_WINDOW_OPENGL | SDL_WINDOW_RESIZABLE | SDL_WINDOW_MAXIMIZED);

    if (!window)
        sdl2_fail("SDL_CreateWindow: ");

    int width, height;
    SDL_GetWindowSize(window, &width, &height);  // получаем разрешение экрана

    SDL_GL_SetAttribute(SDL_GL_CONTEXT_MAJOR_VERSION, 3);
    SDL_GL_SetAttribute(SDL_GL_CONTEXT_MINOR_VERSION, 3);
    SDL_GL_SetAttribute(SDL_GL_CONTEXT_PROFILE_MASK, SDL_GL_CONTEXT_PROFILE_CORE);
    SDL_GL_SetAttribute(SDL_GL_DOUBLEBUFFER, 1);

    SDL_GLContext gl_context = SDL_GL_CreateContext(window);
    if (!gl_context)
        sdl2_fail("SDL_GL_CreateContext: ");

    if (auto result = glewInit(); result != GLEW_NO_ERROR)
        glew_fail("glewInit: ", result);

    if (!GLEW_VERSION_3_3)
        throw std::runtime_error("OpenGL 3.3 is not supported");

    glClearColor(0.8f, 0.8f, 1.f, 0.f);

    GLuint vertex_shader = create_shader(GL_VERTEX_SHADER, vertex_shader_source);
    GLuint fragment_shader = create_shader(GL_FRAGMENT_SHADER, fragment_shader_source);
    GLuint program = create_program(vertex_shader, fragment_shader);  // линкуем шейдеры в итоговую программу

    GLuint vao;
    glGenVertexArrays(1, &vao);
    std::unordered_map<SDL_Keycode, bool> key_down;
    auto last_frame_start = std::chrono::high_resolution_clock::now();


    // === Задание 6 ===
    SDL_GL_SetSwapInterval(0);  // и правда, после добавления этой строки dt стал гораздо меньше

    /*
    // === Задание 1 === 
    GLint scale_id = glGetUniformLocation(program, "scale");  // получаем идентификатор переменной scale внутри программы шейдеров
    glUseProgram(program);  // обязательно приходиться указывать это (то есть что используем программу), чтобы далее передать значение переменной
    glUniform1f(scale_id, 0.5);  // по идентификатору устанавливаем значение переменной = 0.5

    // === Задание 2 ===
    GLint angle_id = glGetUniformLocation(program, "angle");  // аналогично получаем идентификатор angle
    */

    float time = 0.0;  // зануляем текущее время

    // === Задание 3,4 ===
    GLint transform_id = glGetUniformLocation(program, "transform");  // получаем идентификатор матрицы трансформации
    // === Задание 5 ===
    GLint view_id = glGetUniformLocation(program, "view");

    // === Задание 7 ===
    float hex_x = 0.0, hex_y = 0.0;  // текущие координаты центра шестиугольника

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
        case SDL_KEYDOWN:  // в словаре храним true, если соответствующая кнопка (стрелочка на клавиатуре) нажата (в данный момент, на данном кадре отрисовки) и false иначе
            key_down[event.key.keysym.sym] = true;
            break;
        case SDL_KEYUP:
            key_down[event.key.keysym.sym] = false;
            break;
        }

        if (!running)
            break;

        auto now = std::chrono::high_resolution_clock::now();
        float dt = std::chrono::duration_cast<std::chrono::duration<float>>(now - last_frame_start).count();
        last_frame_start = now;

        glClear(GL_COLOR_BUFFER_BIT);

        /*
        // === Задание 6 ===
        dt = 0.016f;  // фиксируем dt -> какой-то эффект есть...
        */

        /*
        // === Задание 2 ===
        time += dt;  // наращиваем время
        glUniform1f(angle_id, time);  // в качестве угла поворота передаём само время (его интерпретируем как меняющийся со врменем угол... так можно, ведь все функии по модулю 360 градусов будут его рассматривать)
        */

        /*
        // === Задание 3 === 
        time += dt;  // наращиваем время, которое будет выступать снова в качестве угла поворота
        float scale = 0.5;  // устанавливаем масштаб
        float angle = time;
        float transform[16] =
        {
        scale * cosf32(angle), -scale * sinf32(angle), scale * 1.0f, 0,
        scale * sinf32(angle), scale * cosf32(angle), scale * 1.0f, 0,
        scale * 0.0f, scale * 0.0f, scale * 1.0f, 0.0,
        0.0, 0.0, 0.0, 1.0
        };  // создаём матрицу преобразований (она 4-ёх мерная, чтобы работать с проективными координатами) -> её общий вид (получается как сначала поворот, а уже затем сдвиг! - они не коммутативны) такой:
            // (R v)  - слева сверху стоит матрица R (3 на 3) поворота (в нашем случае её третий столбец = 1, так как третью координату не трогаем; также все элементы мматрицы умножены на scale для изменения масштаба);
            // (0 1)  - справа сверху стоит вектор v (3 на 1 - сдвиг; в нашем случае сдвига нет (так как хотим вращение вокруг центра треугольника, а он совпадает с 0.0 - началом координат... но можно и дописать сдвиг,
            // тогда треугольник продолжит вращаться вокруг своего центра, но центр подвинется (так как у нас сначала поворот, а затем сдвиг - не наборот! -> если бы наоборот, треугольник стал врщаться вокруг какой-то иной точки, не своего центра)) -> он = 0)
        glUseProgram(program);  // указываем, что используем программу
        glUniformMatrix4fv(transform_id, 1, GL_TRUE, transform);  // после этого передаём матрицу в качестве значение переменной tarnsform_id внутри программы
                                                                  // (обязательно передаём GL_TRUE в параметр transpose, чтобы матрица была транспонирована, так как мы записываем матрицы сначала
                                                                  // по строкам, а библиотека хочет наоборот...)
                                                                  // (параметр 1 видимо указывает на то, что целевая uniform переменная в программе ровно одна - внеё весь этот массив целиком идёт)
        */


        /*
        // === Задание 4 ===
        time += dt;
        float scale = 0.5; 
        float angle = time;
        float x = 0.31 * sinf32(time);  // заводим какие-то переменные сдвига на (x,y), зависящие от времени
        float y = 0.45 * cosf32(time);  // (0.31 и 0.45 - просто так, ничего не значат)
        float transform[16] =
        {
        scale * cosf32(angle), -scale * sinf32(angle), scale * 1.0f, x,  // относительно задания 3 меняется только то, что дописываем сдвиг сюда
        scale * sinf32(angle), scale * cosf32(angle), scale * 1.0f, y,
        scale * 0.0f, scale * 0.0f, scale * 1.0f, 0.0,
        0.0, 0.0, 0.0, 1.0
        };  
        glUseProgram(program); 
        glUniformMatrix4fv(transform_id, 1, GL_TRUE, transform);

        // === Задание 5 ===
        float aspect_ratio = width * 1.0f / height;  // домножаем на 1.0f, чтобы всё привести к float и было нецелочисленное деление
        float view[16] = 
        {
            1.0f/aspect_ratio, 0.0, 0.0, 0.0,
            0.0, 1.0, 0.0, 0.0,
            0.0, 0.0, 1.0, 0.0,
            0.0, 0.0, 0.0, 1.0
        };  // матрица, которая почти единичная, но первую (x-координату) делит на aspect_ratio -> соотношение сторон выравнивается -> треугольник не плющется
        glUniformMatrix4fv(view_id, 1, GL_TRUE, view);  // повторно писать glUseProgram необязательно

        
        glUseProgram(program);
        glBindVertexArray(vao);
        // === Задания до 7 ===
        glDrawArrays(GL_TRIANGLES, 0, 3);  // тут происходит отрисовка вершин треугольника (с использованием установленных шейдеров и их переменных ранее) ->
                                             // -> на каждой итерации (каждом кадре экрана) отрисовка происходит с разным установленным angle (=текущее время) -> так как кадрый
                                             // меняются очень часто (~ 60 раз в секунду), то выглядеть это будет как плавное вращение треугольника
        */

        // === Задание 6 и 7 ===
        std::cout << dt << " ";

        time += dt;
        float scale = 0.5; 
        float angle = time;
        float speed = 1;  // коэффициент скорости смещения шестиугольника
        if (key_down[SDLK_UP])  //  зависимости от того, какие клавиши нажаты, наращиваем соответствующие коордианты центра шестиугольника
            hex_y += speed * dt;
        if (key_down[SDLK_DOWN])
            hex_y -= speed * dt;
        if (key_down[SDLK_LEFT])  // например, пока нажата стрелочка влево, уменьшаем x-координату
            hex_x -= speed * dt;  // (формальнее - если на данном кадре отрисовки всё ещё нажата кнопка влево -> уменьшаем координату; так как частота отрисовки большая, это будет выглядеть как плавное движение по нажатии кнопки влево)
        if (key_down[SDLK_RIGHT])
            hex_x += speed * dt;
        float transform[16] =
        {
        scale * cosf32(angle), -scale * sinf32(angle), scale * 1.0f, hex_x, 
        scale * sinf32(angle), scale * cosf32(angle), scale * 1.0f, hex_y,
        scale * 0.0f, scale * 0.0f, scale * 1.0f, 0.0,
        0.0, 0.0, 0.0, 1.0
        };  // тут всё как в предыдущих заданиях: матрица поврота, чтобы шестиугольник вращался вокруг своего центра + сдвиг, перемещающий центр  
        
        float aspect_ratio = width * 1.0f / height; 
        float view[16] = 
        {
            1.0f/aspect_ratio, 0.0, 0.0, 0.0,
            0.0, 1.0, 0.0, 0.0,
            0.0, 0.0, 1.0, 0.0,
            0.0, 0.0, 0.0, 1.0
        };  

        glUseProgram(program); 
        glUniformMatrix4fv(transform_id, 1, GL_TRUE, transform);
        glUniformMatrix4fv(view_id, 1, GL_TRUE, view);  // устанваливаем матрицы

        glBindVertexArray(vao);
        glDrawArrays(GL_TRIANGLE_FAN, 0, 8);  // пользуемся лекциями, чтобы нарисовать многоугольник с помощью ключа GL_TRIANGLE_FAN

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
