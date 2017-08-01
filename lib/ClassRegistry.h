#ifndef WZANN_CLASSREGISTRY_H_
#define WZANN_CLASSREGISTRY_H_


#include <atomic>
#include <string>
#include <cassert>
#include <unordered_map>

#include <boost/function.hpp>
#include <boost/functional/factory.hpp>


namespace wzann {


    template <class C>
    struct ClassRegistration
    {
        enum { Registered = 0 };
    };


    /*!
     * \brief A simple class registry for object serialization
     *
     * The ClassRegistry is a Singleton class that helps with registering
     * and dynamically creating classes. A class that wants to use this
     * facility needs to call the WINZENT_REGISTER_CLASS() macro in its
     * header file. This makes a struct `ClassRegistration<T>` available,
     * which serves as data object.
     *
     * Creating a new class can be done in two ways: First, by using
     * ClassRegistration<T>::createNew(), which works at compile time.
     * Second, if the name of the class is only known at runtime,
     * ClassRegistry<BaseClass>::create<DerivedClass>(string const& key)
     * can be used to create a new instance of `DerivedClass` from the
     * appropriate ClassRegistry object.
     *
     * To make this work, the public default constructor of the target
     * class must be declared with the `Q_INVOKABLE` macro.
     */
    template <class BaseClass>
    class ClassRegistry
    {
    public:


        /*!
         * \brief The factory method for the default constructor of
         *  `BaseClass`.
         */
        typedef boost::function<BaseClass* ()> Factory;


        typedef std::unordered_map<std::string, Factory> Registry;
        typedef std::pair<
                typename Registry::const_iterator,
                typename Registry::const_iterator> RegistryConstRange;


        /*!
         * \brief Returns the ClassRegistry's singleton instance
         *
         * \return The usable ClassRegistry instance
         */
        static ClassRegistry<BaseClass>* instance()
        {
            if (nullptr == ClassRegistry<BaseClass>::m_instance) {
                ClassRegistry<BaseClass>::m_instance =
                        new ClassRegistry<BaseClass>();
            }

            return ClassRegistry<BaseClass>::m_instance;
        }


        /*!
         * \brief Checks whether a given key is registered
         *
         * \param key The key to check for
         *
         * \return true if the key is registered, false if not.
         */
        bool isRegistered(std::string const& key) const
        {
            return m_registry.find(key) != m_registry.end();
        }


        /*!
         * \brief Introspectional, read-only access to the registry
         *
         * \return A RegistryConstRange over the whole registry
         */
        RegistryConstRange registry() const
        {
            return std::make_pair(m_registry.begin(), m_registry.end());
        }


        /*!
         * \brief Registers a class
         *
         * \param key The class' meta object
         *
         * \return The internal ID
         */
        int registerClass(std::string const& key, Factory const& factory)
        {
            m_registry[key] = factory;
            return m_nextId++;
        }


        /*!
         * \brief Creates a new instance of the registered class
         *
         * \param[in] key The class' name
         *
         * \return The new object instance, as its base class
         */
        BaseClass* create(std::string const& key)
        {
            assert(isRegistered(key));
            return m_registry[key]();
        }


        template <class Derived>
        Derived* create(std::string const& key)
        {
            return reinterpret_cast<Derived*>(create(key));
        }


    private:


        //! Singleton instance
        static ClassRegistry<BaseClass>* m_instance;


        //! \brief The actual name => class mapping registry
        Registry m_registry;


        //! \brief The next ID for a registration
        std::atomic<int> m_nextId;


        //! Private default constructor
        explicit ClassRegistry(): m_nextId(0)
        {
        }
    };


    template <class BaseClass>
    ClassRegistry<BaseClass>* ClassRegistry<BaseClass>::m_instance = nullptr;
} // namespace wzann


#define WZANN_REGISTER_CLASS(KLASS, BaseClass)                              \
namespace wzann {                                                           \
    template <>                                                             \
    struct ClassRegistration<KLASS>                                         \
    {                                                                       \
        enum { Defined = 1 };                                               \
        static const int id;                                                \
                                                                            \
        static const char* className()                                      \
        {                                                                   \
            return #KLASS;                                                  \
        }                                                                   \
    };                                                                      \
                                                                            \
    const int ClassRegistration<KLASS>::id =                                \
            ClassRegistry<BaseClass>::instance()->registerClass(            \
                #KLASS,                                                     \
                []() -> BaseClass* { return new KLASS (); });               \
}

#endif // WZANN_CLASSREGISTRY_H_
