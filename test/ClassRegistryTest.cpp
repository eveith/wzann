#include <gtest/gtest.h>

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
