#pragma once
#include "AbstractTreeGenerator.h"

template <Distance _distance>
class DistanceCalculator : public AbstractTreeGenerator {
private:
	string out_file;
	bool generate_square_matrix;
	bool calculate_pid;
	bool enable_cuda;
public:
	DistanceCalculator(
		int n_threads, 
		instruction_set_t instruction_set, 
		const string& out_file, 
		bool generate_square_matrix, 
		bool calculate_pid,
		bool enable_cuda = false)
		: 
		AbstractTreeGenerator(n_threads, instruction_set), 
		out_file(out_file), 
		generate_square_matrix(generate_square_matrix), 
		calculate_pid(calculate_pid),
		enable_cuda(enable_cuda) {}

protected:
	void run(std::vector<CSequence*>& sequences, tree_structure& tree) override;

};