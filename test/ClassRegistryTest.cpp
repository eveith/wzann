#include <gtest/gtest.h>

#include <boost/range.hpp>

#include "ClassRegistry.h"
#include "ClassRegistryTest.h"


namespace Mock {
    struct BClass
    {
        bool exists() const { return true; }
    };


    struct DClass: public BClass
    {
        int answer() const { return 42; }
    };
}


WZANN_REGISTER_CLASS(Mock::DClass, Mock::BClass)


TEST(ClassRegistryTest, testRegistryIterator)
{
    int iterated = 0;
    auto* cr = wzann::ClassRegistry<Mock::BClass>::instance();

    for (auto const& i : boost::make_iterator_range(cr->registry())) {
        ASSERT_EQ(i.first, "Mock::DClass");
        iterated += 1;
    }

    ASSERT_EQ(iterated, 1);
}


TEST(ClassRegistryTest, testClassCreation)
{
    ASSERT_EQ(
            1,
            wzann::ClassRegistration<Mock::DClass>::Defined);

    Mock::BClass* b = wzann::ClassRegistry<Mock::BClass>::instance()
            ->create("Mock::DClass");
    ASSERT_TRUE(b->exists());
    ASSERT_NE(
            nullptr,
            reinterpret_cast<Mock::DClass*>(b));
    delete b;

    Mock::DClass* d = wzann::ClassRegistry<Mock::BClass>::instance()
            ->create<Mock::DClass>("Mock::DClass");
    ASSERT_TRUE(d->exists());
    ASSERT_EQ(
            42,
            d->answer());
    delete d;
}
