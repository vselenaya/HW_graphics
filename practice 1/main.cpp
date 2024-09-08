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



using namespace std;


GLuint create_shader(GLenum shader_type, const char * shader_source) {
    /*
    Вспомогательная функция для создания шейдера:
        • shader_type – тип создаваемого шейдера
        • shader_source – строка с кодом шейдера
    */

    // создаём пустой объект шейдера нужного типа -> функция возвращает некоторое ненулевое число (id шейдера), по которому к нему можно обрщаться:
    GLuint shader_id = glCreateShader(shader_type);  
    
    // данная функция заменяет в шейдере (доступ к которому происходит по shader_id) его исходный код (так как пока он был пустой, то просто 
    // добавляем к нему код): для этого передаём указатель на массив строк кода (строка типа char* -> массив (=указатель на его начало) типа char**;
    // в данном случае у нас одна строка shader_source -> передаём массив из одной неё - это просто указатель на неё саму), количество элементов
    // в нём (1) и указатель на массив (того же размера, что массив со строками) длин каждой строки с кодом (в данном случае - NULL, что
    // означает, что все строки с кодом ноль-терменированные, то есть оканчиваются на символ с кодом 0):
    glShaderSource(shader_id, 1, &shader_source, NULL);

    // компиллирует код внутри шейдера:
    glCompileShader(shader_id);

    // в переменную param сохраняем информацию, успешно ли скомпиллировался код в шейдере:
    GLint param;
    glGetShaderiv(shader_id, GL_COMPILE_STATUS, &param);

    // если компилляция провалилась:
    if (param != GL_TRUE) {
        GLint len;
        char info_log[1024];
        glGetShaderInfoLog(shader_id, 1024, &len, info_log);  // получаем log об ошибке (записываем в массив info_log макс длины 1024 байт) и его длину len

        throw runtime_error(string("Не скомпилирован shader!\nLog: ") + string(info_log, len));  // бросаем ошибку
    }
    return shader_id;
}


GLuint create_program(GLuint vertex_shader, GLuint fragment_shader) {
    /*
    Функция создания шейдерной программы (Программа = несколько скомпилированных шейдеров, слинкованных вместе).
        • vertex_shader – скомпилированный вершинный шейдер (его id)
        • fragment_shader – аналогично фрагментный шейдер
    */

    GLuint prog_id = glCreateProgram();  // создаём пустую программу -> возвращается её числовой идентификатор, id
    glAttachShader(prog_id, vertex_shader);  // по очереди добавляем в неё все скомпилированные шейдеры - в нашем случае vertex_shader и fragment_shader
    glAttachShader(prog_id, fragment_shader); 
    glLinkProgram(prog_id);  // линкуем вместе все добавленные к этому моменты шейдеры в единую программу

    // получаем результат линковки:
    GLint param;
    glGetProgramiv(prog_id, GL_LINK_STATUS, &param);

    // в случае ошибки линковки снова выводим ошибку:
    if (param != GL_TRUE) {
        char info_log[1024];
        GLint len;
        glGetProgramInfoLog(prog_id, 1024, &len, info_log);

        throw runtime_error(string("Не скомпилирована program!\nLog: ") + string(info_log, len));
    }
    return prog_id;
}


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


int main() try {
    if (SDL_Init(SDL_INIT_VIDEO) != 0)
        sdl2_fail("SDL_Init: ");

    SDL_Window * window = SDL_CreateWindow("Graphics course practice 1",
        SDL_WINDOWPOS_CENTERED,
        SDL_WINDOWPOS_CENTERED,
        800, 600,
        SDL_WINDOW_OPENGL | SDL_WINDOW_RESIZABLE | SDL_WINDOW_MAXIMIZED);  // создаём окошко размера 800 на 600 на экране -> в нём будет происходить всё рисование

    if (!window)
        sdl2_fail("SDL_CreateWindow: ");  // проверяем, что успешно

    SDL_GL_SetAttribute(SDL_GL_CONTEXT_MAJOR_VERSION, 3);
    SDL_GL_SetAttribute(SDL_GL_CONTEXT_MINOR_VERSION, 3);
    SDL_GL_SetAttribute(SDL_GL_CONTEXT_PROFILE_MASK, SDL_GL_CONTEXT_PROFILE_CORE);
    SDL_GL_SetAttribute(SDL_GL_DOUBLEBUFFER, 1);  // здесь настраиваем SDL

    SDL_GLContext gl_context = SDL_GL_CreateContext(window);
    if (!gl_context)
        sdl2_fail("SDL_GL_CreateContext: ");

    if (auto result = glewInit(); result != GLEW_NO_ERROR)
        glew_fail("glewInit: ", result);

    if (!GLEW_VERSION_3_3)
        throw std::runtime_error("OpenGL 3.3 is not supported");

    glClearColor(0.8f, 0.8f, 1.f, 0.f);  // устанавливаем цвет, которым будет очистка окна (в данном случае - голубоватый)

    // =============================================================


    // === Задание 1 и 2: ===
    // create_shader(GL_FRAGMENT_SHADER, "мяу");  // пытаемся скомпиллировать некорректную программу из слова "мяу"


    // === Задание 3 и 4: ===
    /*
    const char fragment_source[] =
    R"(#version 330 core

    layout (location = 0) out vec4 out_color;

    void main() {
        // vec4(R, G, B, A) - настройка цвета в виде вектора из 4-ёх чисел от 0.0 до 1.0: красный (R), зелёный (G), синий (B), прозрачность (A; 0.0 - непрозрачный)
        // каждое число от 0 до 1 задаёт интенсивность соответствующего канала (R, G, B), получая итоговый цвет как смесь стандартных каналов (R, G, B)...
        out_color = vec4(1.0, 0.0, 0.0, 1.0);
    }

    )";  // пишем исходный код фрагментного шейдера с помощью R-строк (они \n проставляют автоматически при переходе на новую строку)

    const char vertex_source[] = 
    R"(#version 330 core

    const vec2 VERTICES[3] = vec2[3](
        vec2(0.0, 0.0),
        vec2(1.0, 0.0),
        vec2(0.0, 1.0)
    );

    void main() {
        gl_Position = vec4(VERTICES[gl_VertexID], 0.0, 1.0);
    }
    
    )";  // и исходный код вершинного
    */

    
    // === Задание 7 и 8 === (дописываем шейдеры)
    /*
    const char fragment_source[] =
    R"(#version 330 core

    layout (location = 0) out vec4 out_color;

    flat in vec3 color;  // переменная, которая пришла из вершинного шейдера (если убрать flat, то будет интерполироваться цвет - иначе использвуется цвет последней вершины)
    void main() {
        // vec4(R, G, B, A)
        out_color = vec4(color, 1.0);  // в качестве цвета (чисел R, G, B) передаём color - вектор из трёх чисел
    }

    )";

    const char vertex_source[] = 
    R"(#version 330 core

    const vec2 VERTICES[3] = vec2[3](  // вектор положений вершин
        vec2(0.0, 0.0),
        vec2(1.0, 0.0),
        vec2(0.0, 1.0)
    );

    flat out vec3 color;  // судя по всему, это переменная, через которую передаётся информация из вершинного (текущего) шейдера во фрагментный
    void main() {
        gl_Position = vec4(VERTICES[gl_VertexID], 0.0, 1.0);
        color = vec3(1-gl_VertexID/3.0, gl_VertexID/3.0, 0);  // тут пишем цвет (как вектор из трёх чисел: R, G, B), зависящий от индексв вершины
    }

    )";
    */


    // === Задание 9* ===
    const char fragment_source[] =
    R"(#version 330 core

    layout (location = 0) out vec4 out_color;

    in vec2 pos;  // получаем координаты (либо вершины треугольника, либо пикселя через интерполяцию)
    void main() {
        vec3 color; 
        int x = int(floor(pos[0] * 10.0));  // координаты прямоугольника в шахматной раскраске
        int y = int(floor(pos[1] * 10.0));
        if ((x + y) % 2 == 0)  // чередуем белый и чёрный
            color = vec3(1.0, 1.0, 1.0);
        else
            color = vec3(0.0, 0.0, 0.0);
        
        out_color = vec4(color, 1.0);  // в качестве цвета (чисел R, G, B) передаём color - вектор из трёх чисел
    
    }

    )";

    const char vertex_source[] = 
    R"(#version 330 core

    const vec2 VERTICES[3] = vec2[3](  // вектор положений вершин
        vec2(0.0, 0.0),
        vec2(1.0, 0.0),
        vec2(0.0, 1.0)
    );

    out vec2 pos;  // выходное значение - координаты вершины (они будут передаваться в фргаментный шейдер для каждой из 3 вершин треугольника,
                   // а для всех остальных пикселей pos будет интерполироваться относительно этих координат...)
    void main() {
        gl_Position = vec4(VERTICES[gl_VertexID], 0.0, 1.0);
        pos = VERTICES[gl_VertexID];
    }

    )";


    // === Задание 5: ===
    auto fragment_shader = create_shader(GL_FRAGMENT_SHADER, fragment_source);
    auto vertex_shader = create_shader(GL_VERTEX_SHADER, vertex_source);
    auto program = create_program(vertex_shader, fragment_shader);  // создаём шейдеры из их кода, затем линкуем в программу


    // === Задание 6: ===
    GLuint vertex_arr_object;
    glGenVertexArrays(1, &vertex_arr_object);  // создаём один (1) Vertex Array Object и сохраняем по адресу vertex_arr_object


    // === начало основного цикла (в этом цикле постоянно что-то отрисовывается внутри окна) ===
    bool running = true;
    while (running) {
        for (SDL_Event event; SDL_PollEvent(&event);) switch (event.type)
        {
        case SDL_QUIT:
            running = false;
            break;
        }

        if (!running)
            break;

        glClear(GL_COLOR_BUFFER_BIT);

        // === Задание 6: ===
        glUseProgram(program);  // включаем использование созданной программы (то есть устанавливаем её как часть текущего состояния рендеринга)
        glBindVertexArray(vertex_arr_object);  // привязываем созданный VAO
        glDrawArrays(GL_TRIANGLES, 0, 3);  // рисуем треугольники: начиная с вершины 0 ровно 3 штуки

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
