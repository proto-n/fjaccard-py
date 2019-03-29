#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>
#include <pybind11/stl.h>
#include <Eigen/Dense>
#include <unordered_map>
#include <utility>
#include <tuple>
#include <map>
#include <set>
#include <vector>
#include <algorithm>

#include "tuple_hash.h"

using MatrixXi = Eigen::Matrix<long, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor>;
using VectorXi = Eigen::Matrix<long, Eigen::Dynamic, 1>;
using MatrixXiRM = Eigen::Matrix<long, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
using MatrixXdRM = Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
using namespace std;

class FJaccard{
protected:
	unordered_map<tuple<long, long>, double> similarity_cache_;
	vector<vector<long>> item_users_;
	int cache_limit_;

	double calculate_jaccard_(const vector<long> &i1v, const vector<long> &i2v){
		std::vector<long> intersection;
		intersection.reserve(min(i1v.size(), i2v.size()));
		set_intersection(
			i1v.begin(), i1v.end(),
			i2v.begin(), i2v.end(),
			std::back_inserter(intersection)
		);
		if(i1v.size() + i2v.size() == 0){
			return 0;
		}
		return double(intersection.size()) / (i1v.size() + i2v.size() - intersection.size());
	}
	void build_userlists_(
		const Eigen::Ref<const MatrixXi> &user_vector,
		const Eigen::Ref<const MatrixXi> &item_vector
	){
		int items_max = 0;
		for(uint i=0; i< item_vector.rows(); i++){
			if(item_vector(i,0) > items_max){
				items_max = item_vector(i,0);
			}
		}
		item_users_ = vector<vector<long>>(items_max+1, vector<long>());
		for(uint i=0; i < item_vector.rows(); i++){
			item_users_[item_vector(i,0)].push_back(user_vector(i,0));
		}
		for(auto &v : item_users_){
			sort(v.begin(), v.end());
		}
	}

public:
	FJaccard(
		const Eigen::Ref<const MatrixXi> &user_vector,
		const Eigen::Ref<const MatrixXi> &item_vector,
		int cache_limit
	) {
		build_userlists_(user_vector, item_vector);
		cache_limit_ = cache_limit;
	}

	double query(long i1, long i2){
		if(i1 == i2){
			return 1;
		}
		if(i1 >= item_users_.size() || i2 >= item_users_.size()){
			return 0;
		}
		bool need_cache = true;
		vector<long> *i1v, *i2v;
		if(cache_limit_ != 0){
			i1v = &item_users_[i1];
			i2v = &item_users_[i2];
			need_cache = i1v->size() >= cache_limit_ && i2v->size() >= cache_limit_;
		}
		if(need_cache){
			auto cache_hit = similarity_cache_.find(make_tuple(min(i1, i2), max(i1, i2)));
			if(cache_hit != similarity_cache_.end()){
				return cache_hit->second;
			}
		}
		if(cache_limit_ == 0){
			i1v = &item_users_[i1];
			i2v = &item_users_[i2];
		}
		double jaccard = calculate_jaccard_(*i1v, *i2v);
		if(need_cache){
			similarity_cache_[make_tuple(min(i1, i2), max(i1, i2))] = jaccard;
		}
		return jaccard;
	}

	MatrixXdRM query_square(const Eigen::Ref<const MatrixXi> &items){
		MatrixXdRM result = MatrixXdRM::Identity(items.rows(), items.rows());
		for(uint i=0; i<items.rows(); i++){
			for(uint j=i+1; j<items.rows(); j++){
				double jaccard = query(items(i,0),items(j,0));
				result(i,j) = jaccard;
				result(j,i) = jaccard;
			}
		}
		return result;
	}

	MatrixXdRM query_pairs(
		const Eigen::Ref<const MatrixXi> &items1,
		const Eigen::Ref<const MatrixXi> &items2
	){
		MatrixXdRM result = MatrixXdRM::Zero(items1.rows(), items2.rows());
		for(uint i=0; i<items1.rows(); i++){
			for(uint j=0; j<items2.rows(); j++){
				result(i,j) = query(items1(i,0),items2(j,0));
			}
		}
		return result;
	}
};


namespace py = pybind11;
PYBIND11_MODULE(fjaccard, m) {
    py::class_<FJaccard>(m, "FJaccard")
        .def(
        	py::init(
        		[](
					const Eigen::Ref<const MatrixXi> &user_vector,
					const Eigen::Ref<const MatrixXi> &item_vector,
					int cache_limit
				) -> unique_ptr<FJaccard> {
					return unique_ptr<FJaccard>(new FJaccard(user_vector, item_vector, cache_limit));
				}
	        ),
			py::arg().noconvert(),
			py::arg().noconvert(),
			py::arg() = 0
		)
		.def("query", &FJaccard::query)
		.def("query_square",
			[](FJaccard &self, const Eigen::Ref<const MatrixXi> &items) -> MatrixXdRM {
				return self.query_square(items);
			},
			py::arg().noconvert()
		)
		.def("query_pairs",
			[](
				FJaccard &self,
				const Eigen::Ref<const MatrixXi> &items1,
				const Eigen::Ref<const MatrixXi> &items2
			) -> MatrixXdRM {
				return self.query_pairs(items1, items2);
			},
			py::arg().noconvert(),
			py::arg().noconvert()
		);

	#ifdef VERSION_INFO
	    m.attr("__version__") = VERSION_INFO;
	#else
	    m.attr("__version__") = "dev";
	#endif
}