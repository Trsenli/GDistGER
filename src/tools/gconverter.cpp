#include <assert.h>

#include <iostream>
#include <utility>
#include <type_traits>

#include "storage.hpp"
#include "type.hpp"
#include "option_helper.hpp"
#include "graph.hpp"

class GConverterOptionHelper : public OptionHelper
{
private:
    args::ValueFlag<std::string> input_path_flag;
    args::ValueFlag<std::string> output_path_flag;
    args::ValueFlag<std::string> static_comp_flag;
public:
    std::string input_path;
    std::string output_path;
    std::string static_comp;
    GConverterOptionHelper() :
        input_path_flag(parser, "input", "input graph path", {'i'}),
        output_path_flag(parser, "output", "output graph path", {'o'}),
        static_comp_flag(parser, "static", "graph type: [weighted | unweighted]", {'s'})
    {
    }

    virtual void parse(int argc, char **argv)
    {
        OptionHelper::parse(argc, argv);

        assert(input_path_flag);
        input_path = args::get(input_path_flag);

        assert(output_path_flag);
        output_path = args::get(output_path_flag);

        assert(static_comp_flag);
        static_comp = args::get(static_comp_flag);
        assert(static_comp.compare("weighted") == 0 || static_comp.compare("unweighted") == 0);
    }
};

void read_txt_edges(const char* input_path, std::vector<Edge<EmptyData> > &edges)
{
    FILE *f = fopen(input_path, "r");
    assert(f != NULL);
    vertex_id_t src, dst;
    while (2 == fscanf(f, "%u %u", &src, &dst))
    {
        edges.push_back(Edge<EmptyData>(src, dst));
    }
    fclose(f);
}

void read_txt_edges(const char* input_path, std::vector<Edge<real_t> > &edges)
{
    FILE *f = fopen(input_path, "r");
    vertex_id_t src, dst;
    real_t weight;
    while (3 == fscanf(f, "%u %u %f", &src, &dst, &weight))
    {
        edges.push_back(Edge<real_t>(src, dst, weight));
    }
    fclose(f);
}

template<typename edge_data_t>
void gconvert(const char* input_path, const char* output_path)
{
    std::vector<Edge<edge_data_t> > edges;
    read_txt_edges(input_path, edges);
    write_graph(output_path, edges.data(), edges.size());
    printf("%zu edges are converted\n", edges.size());
}

int main(int argc, char** argv)
{
    srand(time(NULL));
    GConverterOptionHelper opt;
    opt.parse(argc, argv);
    if (opt.static_comp.compare("weighted") == 0)
    {
        gconvert<real_t>(opt.input_path.c_str(), opt.output_path.c_str());
    } else
    {
        gconvert<EmptyData>(opt.input_path.c_str(), opt.output_path.c_str());
    }
	return 0;
}
