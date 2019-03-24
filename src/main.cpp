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

#include "tuple_hash.h"

using MatrixXi = Eigen::Matrix<long, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor>;
using VectorXi = Eigen::Matrix<long, Eigen::Dynamic, 1>;
using MatrixXiRM = Eigen::Matrix<long, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
using MatrixXdRM = Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
using namespace std;

// template<
//     typename InputIterator1,
//     typename InputIterator2,
//     typename OutputIterator,
//     typename Compare = std::less<>
// >
// auto half_inplace_merge(InputIterator1 first1, InputIterator1 last1,
//                         InputIterator2 first2, InputIterator2 last2,
//                         OutputIterator result, Compare compare={})
//     -> void
// {
//     for (; first1 != last1; ++result) {
//         if (first2 == last2) {
//             std::swap_ranges(first1, last1, result);
//             return;
//         }

//         if (compare(*first2, *first1)) {
//             std::iter_swap(result, first2);
//             ++first2;
//         } else {
//             std::iter_swap(result, first1);
//             ++first1;
//         }
//     }
//     // first2 through last2 are already in the right spot
// }

class FJaccard{
protected:
	unordered_map<tuple<long, long>, double> similarity_cache_;
	vector<vector<long>> item_users_;
	double calculate_jaccard_(long i1, long i2){
		auto &i1v = item_users_[i1];
		auto &i2v = item_users_[i2];
		uint intersection = 0, i=0, j=0;
		while(i < i1v.size() && j < i2v.size()){
			if(i1v[i] == i2v[j]){
				intersection+=1;
				i+=1;
				j+=1;
			} else if(i1v[i] < i2v[j]){
				i++;
			} else {
				j++;
			}
		}
		return double(intersection) / (i1v.size() + i2v.size() - intersection);
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
		const Eigen::Ref<const MatrixXi> &item_vector
	) {
		cout << user_vector.rows() << "x" << user_vector.cols() << endl;
		cout << item_vector.rows() << "x" << item_vector.cols() << endl;
		build_userlists_(user_vector, item_vector);
	}

	double query(long i1, long i2){
		if(i1 == i2){
			return 1;
		}
		auto cache_hit = similarity_cache_.find(make_tuple(i1, i2));
		if(cache_hit != similarity_cache_.end()){
			return cache_hit->second;
		}
		double jaccard = calculate_jaccard_(i1, i2);
		similarity_cache_[make_tuple(i1, i2)] = jaccard;
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

	MatrixXdRM query_pairs(const Eigen::Ref<const MatrixXi> &items1, const Eigen::Ref<const MatrixXi> &items2){
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
					const Eigen::Ref<const MatrixXi> &item_vector
				) {
					return std::unique_ptr<FJaccard>(new FJaccard(user_vector, item_vector));
				}
	        ),
			py::arg().noconvert(),
			py::arg().noconvert()
		)
		.def("query", &FJaccard::query)
		.def(
			"query_square",
			[](FJaccard &self, const Eigen::Ref<const MatrixXi> &items) -> MatrixXdRM {
				return self.query_square(items);
			},
			py::arg().noconvert()
		)
		.def(
			"query_pairs",
			[](FJaccard &self, const Eigen::Ref<const MatrixXi> &items1, const Eigen::Ref<const MatrixXi> &items2) -> MatrixXdRM {
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